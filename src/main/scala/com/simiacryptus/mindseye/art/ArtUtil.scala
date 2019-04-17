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

package com.simiacryptus.mindseye.art

import java.net.URI
import java.util.concurrent.TimeUnit

import com.simiacryptus.mindseye.art.constraints.GramMatrixMatcher
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.line.QuadraticSearch
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy
import com.simiacryptus.mindseye.opt.region.{OrthonormalConstraint, TrustRegion}
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.{StepRecord, TestUtil}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.util.{FastRandom, Util}
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

object ArtUtil {

  def pipelineGraphs(log: NotebookOutput, pipeline: VisionPipeline[VisionPipelineLayer]) = {
    log.subreport(pipeline.name + "_Layers", (sublog: NotebookOutput) => {
      import scala.collection.JavaConverters._
      pipeline.getLayers.keySet().asScala.foreach(layer => {
        sublog.h1(layer.name())
        TestUtil.graph(sublog, layer.getLayer.asInstanceOf[PipelineNetwork])
      })
      null
    })
  }

  def load(log: NotebookOutput, content: Tensor, url: String): Tensor = {
    val noiseRegex = "noise(.*)".r

    def contentDims = content.getDimensions

    url match {
      case "content" => content.copy()
      case "plasma" => Tensor.fromRGB(log.eval(() => {
        new Plasma().paint(contentDims(0), contentDims(1)).toRgbImage
      }))
      case noiseRegex(ampl: String) => Tensor.fromRGB(log.eval(() => {
        new Tensor(contentDims: _*).map((v: Double) => FastRandom.INSTANCE.random() * Option(ampl).filterNot(_.isEmpty).map(Integer.parseInt(_)).getOrElse(100)).toRgbImage
      }))
      case _ => Tensor.fromRGB(log.eval(() => {
        VisionPipelineUtil.load(url, contentDims(0), contentDims(1))
      }))
    }
  }


  def load(log: NotebookOutput, contentDims: Array[Int], url: String): Tensor = {
    val noiseRegex = "noise(.*)".r
    url match {
      case "plasma" => Tensor.fromRGB(log.eval(() => {
        new Plasma().paint(contentDims(0), contentDims(1)).toRgbImage
      }))
      case noiseRegex(ampl: String) => Tensor.fromRGB(log.eval(() => {
        new Tensor(contentDims: _*).map((v: Double) => FastRandom.INSTANCE.random() * Option(ampl).filterNot(_.isEmpty).map(Integer.parseInt(_)).getOrElse(100)).toRgbImage
      }))
      case _ => Tensor.fromRGB(log.eval(() => {
        VisionPipelineUtil.load(url, contentDims(0), contentDims(1))
      }))
    }
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
          case layer: SimpleConvolutionLayer => new OrthonormalConstraint(VisionPipelineUtil.getIndexMap(layer): _*)
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

  def withTrainingMonitor[T](log: NotebookOutput, fn: TrainingMonitor => T) = {
    val history = new ArrayBuffer[StepRecord]
    NotebookRunner.withMonitoredImage(log, () => Util.toImage(TestUtil.plot(history))) {
      val trainingMonitor: TrainingMonitor = getTrainingMonitor(history)
      fn(trainingMonitor)
    }
  }

  def findFiles(key: String, base: String = "s3a://simiacryptus/photos/wikiart/"): Array[String] = {
    val itr = FileSystem.get(new URI(base), VisionPipelineUtil.getHadoopConfig()).listFiles(new Path(base), true)
    val buffer = new ArrayBuffer[String]()
    while (itr.hasNext) {
      val status = itr.next()
      val string = status.getPath.toString
      if (string.contains(key)) buffer += string
    }
    buffer.toArray
  }
}
