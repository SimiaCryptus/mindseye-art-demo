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
import com.simiacryptus.mindseye.art.constraints.{GramMatrixMatcher, RMSContentMatcher}
import com.simiacryptus.mindseye.art.models.Inception5H
import com.simiacryptus.mindseye.art.models.Inception5H._
import com.simiacryptus.mindseye.eval.ArrayTrainable
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

package SimpleStyleTransfer {

  import com.simiacryptus.mindseye.lang.cudnn.CudaMemory

  object EC2 extends SimpleStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

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

  object Local extends SimpleStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
    override def inputTimeoutSeconds = 600
  }

}

abstract class SimpleStyleTransfer extends InteractiveSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 600
  val styleResolution = 1280
  val trainingMinutes: Int = 200
  val trainingIterations: Int = 100

  override def postConfigure(log: NotebookOutput) = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(10000)
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))
    val styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))
    pipelineGraphs(log, Inception5H.getVisionPipeline)
    val network = log.eval(() => {
      val gramMatcher = new GramMatrixMatcher()
      val contentMatcher = new RMSContentMatcher()
      MultiPrecision.setPrecision(SumInputsLayer.combine(
        gramMatcher.build(Inc5H_1a, styleImage),
        gramMatcher.build(Inc5H_2a, styleImage),
        gramMatcher.build(Inc5H_3a, styleImage),
        gramMatcher.build(Inc5H_3b, styleImage),
        contentMatcher.build(contentImage)
      ), Precision.Float).asInstanceOf[PipelineNetwork]
    })
    TestUtil.graph(log, network)
    withMonitoredImage(log, contentImage.toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.eval(() => {
          new IterativeTrainer(new ArrayTrainable(Array[Array[Tensor]](Array(contentImage)), network).setMask(true))
            .setOrientation(new TrustRegionStrategy(new GradientDescent) {
              override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0e-2).setMax(256)
            })
            .setMonitor(trainingMonitor)
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setMaxIterations(trainingIterations)
            .setLineSearchFactory((_: CharSequence) => new BisectionSearch().setCurrentRate(1e4).setSpanTol(1e-4))
            .setTerminateThreshold(java.lang.Double.NEGATIVE_INFINITY)
            .runAndFree
            .asInstanceOf[lang.Double]
        })
      })
    }
  }

}
