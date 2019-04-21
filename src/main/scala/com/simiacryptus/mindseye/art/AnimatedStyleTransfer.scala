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
import com.simiacryptus.mindseye.art.constraints.{ChannelMeanMatcher, GramMatrixEnhancer, GramMatrixMatcher, RMSContentMatcher}
import com.simiacryptus.mindseye.art.models.VGG16._
import com.simiacryptus.mindseye.art.util.ArtSetup
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.{LinearActivationLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.IterativeTrainer
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch
import com.simiacryptus.mindseye.opt.orient.{LBFGS, TrustRegionStrategy}
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object AnimatedStyleTransfer_EC2 extends AnimatedStyleTransfer with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object AnimatedStyleTransfer_Local extends AnimatedStyleTransfer with LocalRunner[Object] with NotebookRunner[Object] {
  override val contentResolution = 512
  override val styleResolution = 512

  override def inputTimeoutSeconds = 5
}

abstract class AnimatedStyleTransfer extends ArtSetup[Object] {

  val contentUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Mandrill_at_SF_Zoo.jpg/1280px-Mandrill_at_SF_Zoo.jpg"
  val inputUrl = contentUrl
  val styleUrl = "https://uploads1.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!HD.jpg"
  val contentResolution = 512
  val styleResolution = 512
  val trainingMinutes: Int = 60
  val trainingIterations: Int = 10
  val tileSize = 400
  val tilePadding = 8
  val maxRate = 1e10

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> AnimatedStyleTransfer.this,
        "style" -> styleLayers.map(_.name())
      ))
    })
    implicit val _log = log
    CudaSettings.INSTANCE().defaultPrecision = precision
    val contentImage = Tensor.fromRGB(log.eval(() => {
      VisionPipelineUtil.load(contentUrl, contentResolution)
    }))
    val contentOperator = new RMSContentMatcher().scale(1e-4)
    val styleOperator = new GramMatrixMatcher().setTileSize(300)
      .combine(new GramMatrixEnhancer().setTileSize(300))
    val colorOperator = new GramMatrixMatcher().setTileSize(300).combine(new ChannelMeanMatcher).scale(1e1)
    val styleGate = new LinearActivationLayer().setScale(1).setBias(0).freeze().asInstanceOf[LinearActivationLayer]
    val styleNetworks = styleLayers.groupBy(_.getPipeline.name).mapValues(pipelineLayers => {
      val pipelineStyleLayers = pipelineLayers.filter(x => styleLayers.contains(x))
      val styleNet = SumInputsLayer.combine((
        pipelineStyleLayers.map(styleOperator.build(_, adjColor(contentImage, Tensor.fromRGB(VisionPipelineUtil.load(styleUrl, styleResolution)))(new NullNotebookOutput()))) ++ List(
          colorOperator.build(adjColor(contentImage, Tensor.fromRGB(VisionPipelineUtil.load(styleUrl, styleResolution)))(new NullNotebookOutput()))
        )): _*)
      styleNet.wrap(styleGate).freeRef()
      styleNet
    })
    val canvas = load(contentImage, inputUrl)
    val trainable = new SumTrainable(styleNetworks.values.map(styleNetwork=>{
      new TiledTrainable(canvas, 300, 32, precision) {
        override protected def getNetwork(regionSelector: Layer): PipelineNetwork = {
          val contentTile = regionSelector.eval(contentImage).getDataAndFree.getAndFree(0)
          regionSelector.freeRef()
          MultiPrecision.setPrecision(SumInputsLayer.combine(
            styleNetwork.addRef(),
            contentOperator.build(VGG16_1c1, contentTile)
          ), precision).freeze().asInstanceOf[PipelineNetwork]
        }
      }
    }).toList: _*)
    val styleCoeffMin = 1e0
    val styleCoeffMax = 2e0
    val steps = 3
    val frames = for (styleCoeff <- Stream.iterate(styleCoeffMin)(_ * Math.pow(styleCoeffMax/styleCoeffMin, 1.0 / steps)).takeWhile(_ <= styleCoeffMax)) yield {
      log.h1(styleCoeff.toString)
      canvas.set(load(contentImage, inputUrl)(new NullNotebookOutput()))
      styleGate.setScale(styleCoeff).setBias(0)
      withMonitoredJpg(() => canvas.toRgbImage) {
        withTrainingMonitor(trainingMonitor => {
          log.eval(() => {
            val search = new ArmijoWolfeSearch().setMaxAlpha(maxRate).setAlpha(maxRate / 10).setRelativeTolerance(1e-3)
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
        })
        null
      }
      contentImage
    }
    NotebookRunner.withMonitoredGif(() => frames.map(_.toRgbImage)) {
      null
    }
  }

  def precision = Precision.Double

  def adjColor(contentImage: Tensor, styleImage: Tensor)(implicit log: NotebookOutput) = {
    colorTransfer(styleImage, List(contentImage), true)
      .copy().freeze().eval(styleImage).getDataAndFree.getAndFree(0)
  }

  def styleLayers: Seq[VisionPipelineLayer] = List(
    //    Inc5H_1a,
    //    Inc5H_2a,
    //    Inc5H_3a,
    //    Inc5H_3b,
    //    Inc5H_4a,
    //    Inc5H_4b,
    //    Inc5H_4c,
    //Inc5H_4d,
    //Inc5H_4e,
    //Inc5H_5a,
    //Inc5H_5b,

    VGG16_0,
    VGG16_1a,
    VGG16_1b1,
    VGG16_1b2,
    VGG16_1c1,
    VGG16_1c2,
    VGG16_1c3,
    VGG16_1d1,
    VGG16_1d2,
    VGG16_1d3,
    VGG16_1e1,
    VGG16_1e2,
    VGG16_1e3
    //    VGG16_2

    //    VGG19_0,
    //    VGG19_1a1,
    //    VGG19_1a2,
    //    VGG19_1b1,
    //    VGG19_1b2,
    //    VGG19_1c1,
    //    VGG19_1c2,
    //    VGG19_1c3,
    //    VGG19_1c4,
    //    VGG19_1d1,
    //    VGG19_1d2,
    //    VGG19_1d3,
    //    VGG19_1d4,
    //    VGG19_1e1,
    //    VGG19_1e2,
    //    VGG19_1e3,
    //    VGG19_1e4
    //    VGG19_2
  )

}
