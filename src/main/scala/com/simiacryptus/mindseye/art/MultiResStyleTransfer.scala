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
import com.simiacryptus.mindseye.art.models.Inception5H._
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.BisectionSearch
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredImage
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object MultiResStyleTransfer_EC2 extends MultiResStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

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

object MultiResStyleTransfer_Local extends MultiResStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
  override val styleResolution = 400
  override val contentResolution = 600
  override val tileSize = 300
  override val trainingIterations: Int = 5

  override def inputTimeoutSeconds = 15
}



abstract class MultiResStyleTransfer extends ArtSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 800
  val styleResolution = 1280
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 100
  val tileSize = 300
  val tilePadding = 8
  val maxRate = 1e5
  val contentCoeff = 1e0

  override def postConfigure(log: NotebookOutput) = {

    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))

    val styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))

    val colorAdjustmentLayer = colorTransfer(log, contentImage, styleImage)

    val trainable_style = log.eval(() => {
      def styleMatcher = new GramMatrixMatcher()

      def getStyleNetwork(styleImage: Tensor) = {
        MultiPrecision.setPrecision(SumInputsLayer.combine(
          styleMatcher.build(Inc5H_2a, styleImage),
          styleMatcher.build(Inc5H_3a, styleImage),
          styleMatcher.build(Inc5H_3b, styleImage),
          styleMatcher.build(Inc5H_4a, styleImage)
        ), Precision.Float).asInstanceOf[PipelineNetwork]
      }

      def getTileTrainer(contentImage: Tensor, styleImage: Tensor, filter: Layer, contentCoeff: Double): TiledTrainable = {
        val styleNetwork = getStyleNetwork(filter.eval(styleImage).getDataAndFree.getAndFree(0))
        new TiledTrainable(contentImage, filter, tileSize, tilePadding) {
          override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
            val contentTile = regionSelector.eval(contentImage).getDataAndFree.getAndFree(0)
            regionSelector.freeRef()
            MultiPrecision.setPrecision(SumInputsLayer.combine(
              styleNetwork.addRef(),
              new RMSContentMatcher().scale(contentCoeff).build(contentTile)
            ), Precision.Float).asInstanceOf[PipelineNetwork]
          }
        }
      }

      new SumTrainable(
        getTileTrainer(contentImage, styleImage, colorAdjustmentLayer.addRef(), contentCoeff / 10),
        getTileTrainer(contentImage, styleImage, PipelineNetwork.wrap(1,
          colorAdjustmentLayer.addRef(),
          new PoolingLayer().setMode(PoolingLayer.PoolingMode.Avg).setWindowXY(2, 2).setStrideXY(2, 2)
        ), contentCoeff),
        getTileTrainer(contentImage, styleImage, PipelineNetwork.wrap(1,
          colorAdjustmentLayer.addRef(),
          new PoolingLayer().setMode(PoolingLayer.PoolingMode.Avg).setWindowXY(3, 3).setStrideXY(2, 2)
        ), contentCoeff)
      )
    })

    withMonitoredImage(log, () => contentImage.toRgbImage) {
      withTrainingMonitor(log, trainingMonitor => {
        log.eval(() => {
          val search = new BisectionSearch().setCurrentRate(maxRate / 10).setMaxRate(maxRate).setSpanTol(1e-1)
          new IterativeTrainer(trainable_style)
            .setOrientation(new TrustRegionStrategy(new GradientDescent) {
              override def getRegionPolicy(layer: Layer) = new RangeConstraint().setMin(0).setMax(256)
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
  }

  def colorTransfer(log: NotebookOutput, contentImage: Tensor, styleImage: Tensor) = {
    val colorAdjustmentLayer = new SimpleConvolutionLayer(1, 1, 3, 3)
    colorAdjustmentLayer.kernel.setByCoord((c: Coordinate) => {
      val coords = c.getCoords()(2)
      if ((coords % 3) == (coords / 3)) 1.0 else 0.0
    })

    val trainable_color = log.eval(() => {
      def styleMatcher = new GramMatrixMatcher() //.combine(new ChannelMeanMatcher().scale(1e0))
      val styleNetwork = MultiPrecision.setPrecision(styleMatcher.build(styleImage), Precision.Float).asInstanceOf[PipelineNetwork]
      new TiledTrainable(contentImage, colorAdjustmentLayer, tileSize, tilePadding) {
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
            .setOrientation(new LBFGS())
            .setMonitor(trainingMonitor)
            .setTimeout(5, TimeUnit.MINUTES)
            .setMaxIterations(20)
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

}
