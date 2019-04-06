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

import java.lang
import java.util.concurrent.TimeUnit

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.ArtUtil._
import com.simiacryptus.mindseye.art.constraints.{ChannelMeanMatcher, GramMatrixMatcher, RMSChannelEnhancer}
import com.simiacryptus.mindseye.art.models.Inception5H
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.BisectionSearch
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

import scala.collection.JavaConversions._
import scala.util.Try

object TextureLayerSurvey_EC2 extends TextureLayerSurvey with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 120

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]",
    "MAX_TOTAL_MEMORY" -> (7.5 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (7.5 * CudaMemory.GiB).toString,
    "MAX_IO_ELEMENTS" -> (1 * CudaMemory.MiB).toString,
    "CONVOLUTION_WORKSPACE_SIZE_LIMIT" -> (256 * CudaMemory.MiB).toString,
    "MAX_FILTER_ELEMENTS" -> (256 * CudaMemory.MiB).toString
  )

}

object TextureLayerSurvey_Local extends TextureLayerSurvey with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 600
  override val contentResolution = 200
  override val styleResolution = 128
}

abstract class TextureLayerSurvey extends InteractiveSetup[Object] {

  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 400
  val styleResolution = 1280
  val trainingMinutes = 200
  val trainingIterations = 10
  val tileSize = 400
  val tilePadding = 16
  val maxRate = 1e6
  val rmsComponent = 1e0
  val rmsGain = 1e-1
  val gramComponent = 1e0

  override def postConfigure(log: NotebookOutput) = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(10000)
    val styleImage: Tensor = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))
    survey(log, Inception5H.getVisionPipeline, styleImage)
    null
  }

  private def survey(log: NotebookOutput, pipeline: VisionPipeline[Inception5H], styleImage: Tensor) = {
    for ((layer: Inception5H) <- pipeline.getLayers.keySet()) {
      log.h2(layer.name())
      val contentImage = Tensor.fromRGB(log.eval(() => {
        val tensor = Plasma.paint(3, 100, 2.1, contentResolution, contentResolution)
        val toRgbImage = tensor.toRgbImage
        tensor.freeRef()
        toRgbImage
      }))
      val operator = new RMSChannelEnhancer().scale(rmsGain)
        .combine(new GramMatrixMatcher().scale(gramComponent))
        .combine(new ChannelMeanMatcher().scale(rmsComponent))
      val styleNetwork: PipelineNetwork = log.eval(() => {
        MultiPrecision.setPrecision(SumInputsLayer.combine(
          operator.build(layer, styleImage)
        ), Precision.Float).asInstanceOf[PipelineNetwork]
      })
      val trainable = new TiledTrainable(contentImage, tileSize, tilePadding) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          styleNetwork.addRef()
        }

        override protected def _free(): Unit = {
          styleNetwork.freeRef()
          super._free()
        }
      }

      withMonitoredImage(log, () => contentImage.toRgbImage) {
        withTrainingMonitor(log, trainingMonitor => {
          Try {
            log.eval(() => {
              val linesearch = new BisectionSearch().setCurrentRate(1e4).setMaxRate(maxRate).setSpanTol(1e-1)
              new IterativeTrainer(trainable)
                .setOrientation(new TrustRegionStrategy(new GradientDescent) {
                  override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0e-2).setMax(256)
                })
                .setMonitor(trainingMonitor)
                .setTimeout(trainingMinutes, TimeUnit.MINUTES)
                .setMaxIterations(trainingIterations)
                .setLineSearchFactory((_: CharSequence) => linesearch)
                .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
                .runAndFree
                .asInstanceOf[lang.Double]
            })
          }
        })
      }
      contentImage.freeRef()
    }
  }
}
