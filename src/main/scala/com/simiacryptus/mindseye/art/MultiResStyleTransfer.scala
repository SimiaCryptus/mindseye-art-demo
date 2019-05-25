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
import com.simiacryptus.mindseye.art.models.Inception5H._
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops.{GramMatrixMatcher, ContentMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, VisionPipelineUtil}
import com.simiacryptus.mindseye.lang.cudnn.{MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Layer, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java.SumInputsLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.{ArmijoWolfeSearch, BisectionSearch}
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object MultiResStyleTransfer_EC2 extends MultiResStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 120

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object MultiResStyleTransfer_Local extends MultiResStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 15
}


abstract class MultiResStyleTransfer extends ArtSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 1024
  val styleResolution = 1024
  val trainingMinutes: Int = 30
  val trainingIterations: Int = 50
  val tileSize = 512
  val tilePadding = 8
  val maxRate = 1e10
  val contentCoeff = 1e-4
  val precision = Precision.Float

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log

    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))

    val styleImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(styleUrl, styleResolution)
    }))

    val colorAdjustmentLayer = colorTransfer(contentImage, styleImage)(log)

    val trainable_style = log.eval(() => {
      def styleOperator = new GramMatrixMatcher()

      def getStyleNetwork(styleImage: Tensor) = {
        MultiPrecision.setPrecision(SumInputsLayer.combine(
          styleOperator.build(Inc5H_1a, styleImage),
          styleOperator.build(Inc5H_2a, styleImage),
          styleOperator.build(Inc5H_3b, styleImage),
          styleOperator.build(VGG19_1a, styleImage),
          styleOperator.build(VGG19_1b1, styleImage),
          styleOperator.build(VGG19_1c1, styleImage)
        ), precision).asInstanceOf[PipelineNetwork]
      }

      def getTileTrainer(contentImage: Tensor, styleImage: Tensor, filter: Layer, contentCoeff: Double): TiledTrainable = {
        val styleNetwork = getStyleNetwork(filter.eval(styleImage).getDataAndFree.getAndFree(0))
        new TiledTrainable(contentImage, filter, tileSize, tilePadding, precision) {
          override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
            val contentTile = regionSelector.eval(contentImage).getDataAndFree.getAndFree(0)
            regionSelector.freeRef()
            MultiPrecision.setPrecision(SumInputsLayer.combine(
              styleNetwork.addRef(),
              new ContentMatcher().scale(contentCoeff).build(VGG19_1c1, contentTile)
            ), precision).asInstanceOf[PipelineNetwork]
          }
        }
      }

      new SumTrainable(
        //        getTileTrainer(contentImage, styleImage, colorAdjustmentLayer.addRef(), contentCoeff / 10),
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

    withMonitoredJpg(() => contentImage.toRgbImage) {
      withTrainingMonitor(trainingMonitor => {
        log.eval(() => {
          val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-1)
          new IterativeTrainer(trainable_style)
            .setOrientation(new TrustRegionStrategy(new LBFGS) {
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
      null
    }
  }

  def colorTransfer(contentImage: Tensor, styleImage: Tensor)(implicit log: NotebookOutput): SimpleConvolutionLayer = {
    val colorAdjustmentLayer = new SimpleConvolutionLayer(1, 1, 3, 3)
    colorAdjustmentLayer.kernel.setByCoord((c: Coordinate) => {
      val coords = c.getCoords()(2)
      if ((coords % 3) == (coords / 3)) 1.0 else 0.0
    })

    val trainable_color = log.eval(() => {
      def styleMatcher = new GramMatrixMatcher() //.combine(new ChannelMeanMatcher().scale(1e0))
      val styleNetwork = MultiPrecision.setPrecision(styleMatcher.build(styleImage), precision).asInstanceOf[PipelineNetwork]
      new TiledTrainable(contentImage, colorAdjustmentLayer, tileSize, tilePadding) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          regionSelector.freeRef()
          styleNetwork.addRef()
        }
      }.setMutableCanvas(false)
    })

    withMonitoredJpg(() => colorAdjustmentLayer.eval(contentImage).getDataAndFree.getAndFree(0).toRgbImage) {
      withTrainingMonitor(trainingMonitor => {
        log.eval(() => {
          val search = new BisectionSearch().setCurrentRate(1e0).setMaxRate(1e3).setSpanTol(1e-3)
          new IterativeTrainer(trainable_color)
            .setOrientation(new LBFGS())
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

}
