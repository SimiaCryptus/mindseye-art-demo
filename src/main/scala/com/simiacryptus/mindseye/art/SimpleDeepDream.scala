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
import com.simiacryptus.mindseye.art.constraints.RMSChannelEnhancer
import com.simiacryptus.mindseye.art.models.Inception5H
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
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

object SimpleDeepDream_EC2 extends SimpleDeepDream with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 120

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

}

object SimpleDeepDream_Local extends SimpleDeepDream with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 15
}

abstract class SimpleDeepDream extends ArtSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val contentResolution = 512
  val trainingMinutes: Int = 30
  val trainingIterations: Int = 15
  val tileSize = 512

  override def postConfigure(log: NotebookOutput) = {
    var contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))
    val trainable: Trainable = log.eval(() => {
      def precision = Precision.Float

      val channelEnhancer = new RMSChannelEnhancer()
      val network = MultiPrecision.setPrecision(SumInputsLayer.combine(
        channelEnhancer.build(Inception5H.Inc5H_4e, contentImage),
        channelEnhancer.build(Inception5H.Inc5H_5a, contentImage),
        channelEnhancer.build(Inception5H.Inc5H_5b, contentImage)
      ), precision).asInstanceOf[PipelineNetwork]
      new TiledTrainable(contentImage.copy(), new PipelineNetwork(1), tileSize, 16, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          network.addRef()
        }
      }
    })
    withMonitoredImage(log, () => contentImage.copy().toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.eval(() => {
          val search = new BisectionSearch().setCurrentRate(1e0).setSpanTol(1e-1)
          IterativeTrainer.wrap(trainable)
            .setOrientation(new TrustRegionStrategy(new GradientDescent) {
              override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0).setMax(255)
            })
            .setMonitor(trainingMonitor)
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setMaxIterations(trainingIterations)
            .setLineSearchFactory((_: CharSequence) => search)
            .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
            .runAndFree
            .asInstanceOf[lang.Double]
        })
      })
    }
    null
  }
}
