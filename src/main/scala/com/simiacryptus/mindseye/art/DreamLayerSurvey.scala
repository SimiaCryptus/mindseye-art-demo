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
import com.simiacryptus.mindseye.art.constraints.{GramMatrixMatcher, RMSChannelEnhancer}
import com.simiacryptus.mindseye.art.models.{Inception5H, VGG16, VGG19}
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

import scala.collection.JavaConversions._
import scala.util.Try

object DreamLayerSurvey_EC2 extends DreamLayerSurvey with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 120

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object DreamLayerSurvey_Local extends DreamLayerSurvey with LocalRunner[Object] with NotebookRunner[Object] {
  override val contentResolution = 256
  override def inputTimeoutSeconds = 600
}

abstract class DreamLayerSurvey extends ArtSetup[Object] {
  val imageUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val contentResolution = 512
  val trainingMinutes = 30
  val trainingIterations = 20
  val tileSize = 512
  val tilePadding = 8
  val maxRate = 1e6
  def precision = Precision.Double

  override def postConfigure(log: NotebookOutput) = {
    survey(log, Inception5H.getVisionPipeline)
    survey(log, VGG19.getVisionPipeline)
    survey(log, VGG16.getVisionPipeline)
    null
  }

  def survey(log: NotebookOutput, pipeline: VisionPipeline[_ <: VisionPipelineLayer]): Unit = {
    log.h1(pipeline.name)
    for (layer <- pipeline.getLayers.keySet()) {
      log.h2(layer.name())
      TestUtil.graph(log, layer.getLayer.asInstanceOf[PipelineNetwork])
      survey(log, layer)
    }
  }

  def survey(log: NotebookOutput, layer: VisionPipelineLayer): Unit = {
    val contentImage = Tensor.fromRGB(VisionPipelineUtil.load(imageUrl, contentResolution))
    val operator = new RMSChannelEnhancer
    val styleNetwork: PipelineNetwork = log.eval(() => {
      MultiPrecision.setPrecision(SumInputsLayer.combine(
        operator.build(layer, contentImage)
      ), precision).freeze().asInstanceOf[PipelineNetwork]
    })
    val trainable = new TiledTrainable(contentImage, tileSize, tilePadding, precision) {
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
            val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-1)
            new IterativeTrainer(trainable)
              .setOrientation(new TrustRegionStrategy(new LBFGS) {
                override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0e-2).setMax(256)
              })
              .setMonitor(trainingMonitor)
              .setTimeout(trainingMinutes, TimeUnit.MINUTES)
              .setMaxIterations(trainingIterations)
              .setLineSearchFactory((_: CharSequence) => search)
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