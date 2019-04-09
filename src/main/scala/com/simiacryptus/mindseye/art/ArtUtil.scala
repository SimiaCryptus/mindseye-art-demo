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

import java.util.concurrent.TimeUnit

import com.simiacryptus.mindseye.art.constraints.GramMatrixMatcher
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.line.BisectionSearch
import com.simiacryptus.mindseye.opt.orient.GradientDescent
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.mindseye.test.{StepRecord, TestUtil}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.util.{FastRandom, Util}

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

  def load(log: NotebookOutput, contentImage: Tensor, url: String): Tensor = {
    load(log, contentImage.getDimensions(), url)
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

  def colorTransfer(log: NotebookOutput, contentImage: Tensor, styleImage: Tensor, tileSize: Int, tilePadding: Int, precision: Precision) = {
    val colorAdjustmentLayer = new SimpleConvolutionLayer(1, 1, 3, 3) //.setPrecision(precision)
    colorAdjustmentLayer.kernel.setByCoord((c: Coordinate) => {
      val coords = c.getCoords()(2)
      if ((coords % 3) == (coords / 3)) 1.0 else 0.0
    })

    val trainable_color = log.eval(() => {
      def styleMatcher = new GramMatrixMatcher() //.combine(new ChannelMeanMatcher().scale(1e0))
      val styleNetwork = MultiPrecision.setPrecision(styleMatcher.build(styleImage), precision).asInstanceOf[PipelineNetwork]
      new TiledTrainable(contentImage, colorAdjustmentLayer, tileSize, tilePadding, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          styleNetwork.addRef()
        }
      }.setMutableCanvas(false)
    })

    withMonitoredImage(log, () => colorAdjustmentLayer.eval(contentImage).getDataAndFree.getAndFree(0).toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.eval(() => {
          val search = new BisectionSearch().setCurrentRate(1e0).setMaxRate(1e3).setSpanTol(1e-3)
          new IterativeTrainer(trainable_color)
            .setOrientation(new GradientDescent())
            .setMonitor(trainingMonitor)
            .setTimeout(5, TimeUnit.MINUTES)
            .setMaxIterations(5)
            .setLineSearchFactory((_: CharSequence) => search)
            .setTerminateThreshold(0)
            .runAndFree
          colorAdjustmentLayer.freeze()
          colorAdjustmentLayer.getJson()
        })
      })
    }
    colorAdjustmentLayer
  }

  def withTrainingMonitor[T](log: NotebookOutput, fn: TrainingMonitor => T) = {
    val history = new ArrayBuffer[StepRecord]
    NotebookRunner.withMonitoredImage(log, () => Util.toImage(TestUtil.plot(history))) {
      val trainingMonitor = new TrainingMonitor() {
        override def clear(): Unit = {
          super.clear()
        }

        override def log(msg: String): Unit = {
          System.out.println(msg)
          super.log(msg)
        }

        override def onStepComplete(currentPoint: Step): Unit = {
          history += new StepRecord(currentPoint.point.getMean, currentPoint.time, currentPoint.iteration)
          super.onStepComplete(currentPoint)
        }
      }
      fn(trainingMonitor)
    }
  }

}
