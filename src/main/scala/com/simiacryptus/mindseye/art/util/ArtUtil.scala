/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.art.util

import java.awt.image.BufferedImage
import java.io.File
import java.net.URI
import java.util
import java.util.UUID
import java.util.concurrent.TimeUnit

import com.simiacryptus.mindseye.art._
import com.simiacryptus.mindseye.art.ops.GramMatrixMatcher
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer
import com.simiacryptus.mindseye.network.{DAGNetwork, PipelineNetwork}
import com.simiacryptus.mindseye.opt.line.QuadraticSearch
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy
import com.simiacryptus.mindseye.opt.region.{OrthonormalConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.{StepRecord, TestUtil}
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook._
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.util.Util
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object ArtUtil {

  private val pngCache = new mutable.HashMap[String, Try[File]]()

  def userSelect(paintings: Array[String])(implicit log: NotebookOutput): HtmlQuery[Map[String, Boolean]] = {
    val ids = paintings.map(_ -> UUID.randomUUID().toString).toMap
    new FormQuery[Map[String, Boolean]](log.asInstanceOf[MarkdownNotebookOutput]) {

      override def height(): Int = 800

      override protected def getFormInnerHtml: String = {
        (for ((url, v) <- getValue.toList.filter(_._2)) yield {
          val filename = url.split('/').last.toLowerCase.stripSuffix(".png") + ".png"
          if (pngCache.getOrElseUpdate(url, Try {
            log.pngFile(ImageArtUtil.load(log, url, 256), new File(log.getResourceDir, filename))
          }).isSuccess) {
            s"""<input type="checkbox" name="${ids(url)}" value="true"><img src="etc/$filename" alt="$url"><br/>"""
          } else {
            ""
          }
        }).mkString("\n")
      }

      override def valueFromParams(parms: util.Map[String, String], files: util.Map[String, String]): Map[String, Boolean] = {
        (for ((k, v) <- getValue) yield {
          k -> parms.getOrDefault(ids(k), "false").toBoolean
        })
      }
    }.setValue(paintings.map(_ -> true).toMap).print()
  }

  def setPrecision(trainable: Trainable, precision: Precision): Boolean = {
    if (CudaSettings.INSTANCE().defaultPrecision == precision) {
      false
    } else {
      CudaSettings.INSTANCE().defaultPrecision = precision
      resetPrecision(trainable, precision)
      true
    }
  }

  def resetPrecision(trainable: Trainable, precision: Precision) = {
    trainable match {
      case trainable: SumTrainable =>
        for (layer <- trainable.getInner) setPrecision(layer, precision)
      case trainable: TiledTrainable =>
        trainable.setPrecision(precision)
      case trainable: Trainable if null != trainable.getLayer =>
        MultiPrecision.setPrecision(trainable.getLayer.asInstanceOf[DAGNetwork], precision)
    }
  }

  def pipelineGraphs(pipeline: VisionPipeline[VisionPipelineLayer])(implicit log: NotebookOutput) = {
    log.subreport(pipeline.name + "_Layers", (sublog: NotebookOutput) => {
      pipeline.getLayers().keys.foreach(layer => {
        sublog.h1(layer.name())
        TestUtil.graph(sublog, layer.getLayer.asInstanceOf[PipelineNetwork])
      })
      null
    })
  }

  def cyclicalAnimation(canvases: => Seq[Tensor]): Seq[BufferedImage] = {
    val list = canvases.filter(_ != null).map(_.toRgbImage)
    if (list.isEmpty) {
      Seq.empty
    } else {
      val maxWidth = list.map(_.getWidth).max
      cyclical(list.map(ImageUtil.resize(_, maxWidth, true)))
    }
  }

  def cyclical[T](list: Seq[T]) = {
    (list ++ list.reverse.tail.dropRight(1))
  }

  def load(content: Tensor, url: String)(implicit log: NotebookOutput): Tensor = {
    load(content.getDimensions, url)
  }

  def load(contentDims: Array[Int], url: String)(implicit log: NotebookOutput = new NullNotebookOutput()): Tensor = {
    ImageArtUtil.loadTensor(log, url, contentDims(0), contentDims(1))
  }

  def colorTransfer
  (
    contentImage: Tensor,
    styleImages: Seq[Tensor],
    orthogonal: Boolean = true,
    tileSize: Int = 400,
    tilePadding: Int = 0,
    precision: Precision = Precision.Float
  )(implicit log: NotebookOutput): Layer = {
    colorTransfer(contentImage, styleImages, tileSize, tilePadding, precision, PipelineNetwork.wrap(1,
      new SimpleConvolutionLayer(1, 1, 3, 3).set((c: Coordinate) => {
        val coords = c.getCoords()(2)
        if ((coords % 3) == (coords / 3)) 1.0 else 0.0
      }),
      new ImgBandBiasLayer(3).setWeights((i: Int) => 0.0)
    )).freeze()
  }

  def colorTransfer(contentImage: Tensor, styleImages: Seq[Tensor], tileSize: Int, tilePadding: Int, precision: Precision, colorAdjustmentLayer: PipelineNetwork): Layer = {
    def styleMatcher = new GramMatrixMatcher() //.combine(new ChannelMeanMatcher().scale(1e0))
    val styleNetwork = MultiPrecision.setPrecision(styleMatcher.build(styleImages: _*), precision).asInstanceOf[PipelineNetwork]
    val trainable_color = new TiledTrainable(contentImage, colorAdjustmentLayer, tileSize, tilePadding, precision) {
      override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
        regionSelector.freeRef()
        styleNetwork.addRef()
      }
    }.setMutableCanvas(false)

    val trainingMonitor = getTrainingMonitor(verbose = false)
    val search = new QuadraticSearch().setCurrentRate(1e0).setMaxRate(1e3).setRelativeTolerance(1e-2)
    new IterativeTrainer(trainable_color)
      .setOrientation(new TrustRegionStrategy() {
        override def getRegionPolicy(layer: Layer): TrustRegion = layer match {
          case null => null
          case layer if layer.isFrozen => null
          case layer: SimpleConvolutionLayer => new OrthonormalConstraint(ImageArtUtil.getIndexMap(layer): _*)
          case _ => null
        }
      })
      .setMonitor(trainingMonitor)
      .setTimeout(5, TimeUnit.MINUTES)
      .setMaxIterations(3)
      .setLineSearchFactory((_: CharSequence) => search)
      .setTerminateThreshold(0)
      .runAndFree
    colorAdjustmentLayer
  }

  def getTrainingMonitor[T](history: ArrayBuffer[StepRecord] = new ArrayBuffer[StepRecord], verbose: Boolean = true): TrainingMonitor = {
    val trainingMonitor = new TrainingMonitor() {
      override def clear(): Unit = {
        super.clear()
      }

      override def log(msg: String): Unit = {
        if (verbose) System.out.println(msg)
        super.log(msg)
      }

      override def onStepComplete(currentPoint: Step): Unit = {
        history += new StepRecord(currentPoint.point.getMean, currentPoint.time, currentPoint.iteration)
        super.onStepComplete(currentPoint)
      }
    }
    trainingMonitor
  }

  def imageGrid(currentImage: BufferedImage, columns: Int = 2, rows: Int = 2) = Option(currentImage).map(tensor => {
    val assemblyLayer = new ImgTileAssemblyLayer(columns, rows)
    val grid = assemblyLayer.eval(Stream.continually(Tensor.fromRGB(tensor)).take(columns * rows): _*)
      .getDataAndFree.getAndFree(0).toRgbImage
    assemblyLayer.freeRef()
    grid
  }).orNull

  def withTrainingMonitor[T](fn: TrainingMonitor => T)(implicit log: NotebookOutput): T = {
    val history = new ArrayBuffer[StepRecord]
    NotebookRunner.withMonitoredJpg(() => Util.toImage(TestUtil.plot(history))) {
      fn(getTrainingMonitor(history))
    }
  }

  def findFiles(key: String, base: String): Array[String] = findFiles(Set(key), base)

  def findFiles(key: String): Array[String] = findFiles(Set(key))

  def findFiles(key: Set[String], base: String = "s3a://data-cb03c/crawl/wikiart/", minSize: Int = 32 * 1024): Array[String] = {
    val itr = FileSystem.get(new URI(base), ImageArtUtil.getHadoopConfig()).listFiles(new Path(base), true)
    val buffer = new ArrayBuffer[String]()
    while (itr.hasNext) {
      val status = itr.next()
      lazy val string = status.getPath.toString
      if (status.getLen > minSize && key.find(string.contains(_)).isDefined) buffer += string
    }
    buffer.toArray
  }

}
