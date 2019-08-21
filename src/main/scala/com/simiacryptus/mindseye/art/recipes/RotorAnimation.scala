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

package com.simiacryptus.mindseye.art.recipes

import java.net.URI
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops.{GramMatrixEnhancer, MomentMatcher}
import com.simiacryptus.mindseye.art.util.ArtSetup.{ec2client, s3client}
import com.simiacryptus.mindseye.art.util.ArtUtil.cyclicalAnimation
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.{BoundedActivationLayer, ImgViewLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.opt.region.RangeConstraint
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, NotebookRunner}

import scala.collection.immutable

object RotorAnimationEC2 extends RotorAnimation with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override val s3bucket = "www.tigglegickle.com"

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

  override def inputTimeoutSeconds = 30

  override def maxHeap: Option[String] = Option("50g")

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (15 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (15 * CudaMemory.GiB).toString
  )

}

object RotorAnimation extends RotorAnimation with LocalRunner[Object] with NotebookRunner[Object]

class RotorAnimation extends RotorArt {

  override lazy val rotationalChannelPermutation: Array[Int] = Array(1, 2, 3)
  override val rotationalSegments: Int = 5
  val colorUrl =
  //    "https://upload.wikimedia.org/wikipedia/commons/3/34/Camp_fire.jpg"
    ""
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_468243743.jpg" // Leaves
  val patternUrl =
  //  ""
    "file:///C:/Users/andre/Downloads/pictures/Pikachu-Pokemon-Wallpapers-SWA0039152.jpg"
  //  "file:///C:/Users/andre/Downloads/pictures/1920x1080-kaufman_63748_5.jpg"
  val styleUrl =
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_1060865300.jpg" // Plasma Ball
    "file:///C:/Users/andre/Downloads/pictures/1920x1080-kaufman_63748_5.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/the-starry-night.jpg"
  //      "file:///C:/Users/andre/Downloads/pictures/shutterstock_240121861.jpg" // Grafiti
  val initUrl: String = "50+noise50"
  val s3bucket: String = "www.tigglegickle.com"
  val transitions = 3

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    log.setArchiveHome(URI.create(s"s3://$s3bucket/${getClass.getSimpleName.stripSuffix("$")}/${UUID.randomUUID()}/"))
    log.onComplete(() => upload(log): Unit)
    log.out(log.jpg(ImageArtUtil.load(log, colorUrl, 600), "Input Color"))
    log.out(log.jpg(ImageArtUtil.load(log, styleUrl, 600), "Input Style"))
    log.out(log.jpg(ImageArtUtil.load(log, patternUrl, 600), "Input Pattern"))
    val canvases: immutable.Seq[AtomicReference[Tensor]] = (1 to (transitions * 2 + 1)).map(_ => new AtomicReference[Tensor](null)).toList
    val renderingFn: Seq[Int] => PipelineNetwork = dims => {
      val network = getKaleidoscope(dims.toArray).copyPipeline()
      network.wrap(new BoundedActivationLayer().setMinValue(0).setMaxValue(255))
      network
    }

    def renderedCanvases = canvases.map(_.get()).filter(_ != null).map(tensor => {
      val network = renderingFn(tensor.getDimensions)
      val transformed = network.eval(tensor).getDataAndFree.getAndFree(0)
      network.freeRef()
      transformed
    })

    val registration = registerWithIndexGIF2(renderedCanvases)
    try {
      val calcFn: Seq[Int] => PipelineNetwork = dims => {
        val padding = 0 // Math.min(256, Math.max(16, dims(0) / 2))
        val viewLayer = renderingFn(dims).copyPipeline()

        if (padding > 0) viewLayer.wrap(new ImgViewLayer(dims(0) + padding, dims(1) + padding, true)
          .setOffsetX(-padding / 2).setOffsetY(-padding / 2)).freeRef()
        viewLayer
      }
      NotebookRunner.withMonitoredGif(() => {
        cyclicalAnimation(renderedCanvases)
      }) {
        log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
          paintBisection("", initUrl, canvases, sub.eval(() => {
            (1 to (transitions * 2 + 1)).map(step => f"step = $step%d" -> {
              var visualStyle: VisualNetwork = new VisualStyleNetwork(
                styleLayers = List(
                  VGG19_1b1,
                  VGG19_1b2,
                  VGG19_1c1,
                  VGG19_1c2,
                  VGG19_1c3,
                  VGG19_1c4
                  //                ).flatMap(baseLayer => List(
                  //                  baseLayer,
                  //                  baseLayer.prependAvgPool(2),
                  //                  baseLayer.prependAvgPool(3)
                  //                )
                ),
                styleModifiers = List(
                  new GramMatrixEnhancer(),
                  new MomentMatcher()
                ),
                styleUrl = List(styleUrl),
                magnification = 8,
                viewLayer = calcFn
              )
              if (null != patternUrl && !patternUrl.isEmpty) visualStyle = visualStyle + new VisualStyleNetwork(
                styleLayers = List(
                  VGG19_1d1,
                  VGG19_1d2,
                  VGG19_1d3,
                  VGG19_1d4,
                  VGG19_1e1,
                  VGG19_1e2,
                  VGG19_1e3,
                  VGG19_1e4
                ).flatMap(baseLayer => List(
                  baseLayer,
                  baseLayer.prependAvgPool(2),
                  baseLayer.prependAvgPool(3)
                )
                ),
                styleModifiers = List(
                  new GramMatrixEnhancer().setMinMax(-1e1, 1e1),
                  new MomentMatcher()
                ),
                styleUrl = List(patternUrl),
                magnification = 8,
                viewLayer = calcFn
              )
              if (null != colorUrl && !colorUrl.isEmpty) visualStyle = visualStyle + new VisualStyleNetwork(
                styleLayers = List(
                  VGG19_0a
                ),
                styleModifiers = List(
                  new GramMatrixEnhancer(),
                  new MomentMatcher().scale(1e1)
                ),
                styleUrl = List(colorUrl),
                magnification = 8,
                viewLayer = calcFn
              )
              visualStyle
            })
          }), new BasicOptimizer {
            override val trainingMinutes: Int = 60
            override val trainingIterations: Int = 30
            override val maxRate = 1e9

            override def renderingNetwork(dims: Seq[Int]): PipelineNetwork = renderingFn(dims)

            override def trustRegion(layer: Layer): RangeConstraint = null
          }, renderingFn, transitions, new GeometricSequence {
            override val min: Double = 240
            override val max: Double = 1024
            override val steps = 4
          }.toStream: _*)(sub)
          null
        })
      }(log)
    } finally {
      registration.foreach(_.stop()(s3client, ec2client))
    }

  }

}
