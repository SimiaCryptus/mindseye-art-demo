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
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtSetup._
import com.simiacryptus.mindseye.art.util.ArtUtil.cyclicalAnimation
import com.simiacryptus.mindseye.art.util.{BasicOptimizer, _}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredGif
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

import scala.collection.immutable

object StyleAnimationEC2 extends StyleAnimation with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override val styleUrl: String = "s3://simiacryptus/photos/shutterstock_240121861.jpg"
  override val contentUrl: String = "s3://simiacryptus/photos/IMG_20181107_171439630_crop.jpg"
  //override val initUrl: String = contentUrl
  override val s3bucket = "www.tigglegickle.com"

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

  override def inputTimeoutSeconds = 30

  override def maxHeap: Option[String] = Option("50g")

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (15 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (15 * CudaMemory.GiB).toString
  )

}

object StyleAnimation extends StyleAnimation with LocalRunner[Object] with NotebookRunner[Object]

class StyleAnimation extends ArtSetup[Object] {
  val contentUrl =
  //    "file:///C:/Users/andre/Downloads/IMG_20170507_162514668.jpg" // Road to city
  //      "file:///C:/Users/andre/Downloads/pictures/E2-E.jpg" // Daddys girl
    "file:///C:/Users/andre/Downloads/pictures/IMG_20181107_171439630_crop.jpg" // Boy portrait
  //    "file:///C:/Users/andre/Downloads/img11262015_0645_2.jpg" // Kids by the lake

  val styleUrl =
  //    "file:///C:/Users/andre/Downloads/pictures/the-starry-night.jpg"
  //      "file:///C:/Users/andre/Downloads/pictures/shutterstock_240121861.jpg" // Grafiti
    "file:///C:/Users/andre/Downloads/pictures/1920x1080-kaufman_63748_5.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/Pikachu-Pokemon-Wallpapers-SWA0039152.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_1060865300.jpg" // Plasma Ball
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_468243743.jpg" // Leaves

  val initUrl: String = "50+noise50" //contentUrl
  val s3bucket: String = ""
  val transitions = 2

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    log.setArchiveHome(URI.create(s"s3://$s3bucket/${getClass.getSimpleName.stripSuffix("$")}/${UUID.randomUUID()}/"))
    log.onComplete(() => upload(log): Unit)

    log.out(log.jpg(ImageArtUtil.load(log, styleUrl, 600), "Input Style"))
    log.out(log.jpg(ImageArtUtil.load(log, contentUrl, 600), "Reference Content"))
    val canvases: immutable.Seq[AtomicReference[Tensor]] = (1 to numSteps).map(_ => new AtomicReference[Tensor](null)).toList
    val registration = registerWithIndexGIF_Cyclic(canvases.map(_.get()))
    try {
      withMonitoredGif(() => cyclicalAnimation(canvases.map(_.get()))) {
        log.subreport("Painting", (sub: NotebookOutput) => {
          paintBisection(contentUrl, initUrl, canvases, sub.eval(() => {
            (1 to numSteps).map(step => s"step $step" -> {
              new VisualStyleNetwork(
                styleLayers = List(
                  VGG19_1a,
                  VGG19_1b1,
                  VGG19_1b2,
                  VGG19_1c1,
                  VGG19_1c2,
                  VGG19_1c3,
                  VGG19_1c4,
                  VGG19_1d1,
                  VGG19_1d2,
                  VGG19_1d3,
                  VGG19_1d4
                ).flatMap(baseLayer => List(
                  baseLayer,
                  baseLayer.prependAvgPool(2),
                  baseLayer.prependAvgPool(3)
                )),
                styleModifiers = List(
                  new GramMatrixEnhancer().setMinMax(0.5, 0.5),
                  new MomentMatcher()
                ),
                styleUrl = List(styleUrl),
                magnification = 2
              ).withContent(List(
                VGG19_1b2,
                VGG19_1c2,
                VGG19_1c4,
                VGG19_1d2,
                VGG19_1d4
              ).flatMap(baseLayer => List(
                baseLayer,
                baseLayer.prependAvgPool(2),
                baseLayer.prependAvgPool(3)
              )), List(
                new ContentMatcher().scale(1e0)
              ))
            })
          }), new BasicOptimizer {
            override val trainingMinutes: Int = 90
            override val trainingIterations: Int = 20
            override val maxRate = 1e9
          }, _ => new PipelineNetwork(1), transitions, new GeometricSequence {
            override val min: Double = 400
            override val max: Double = 1024
            override val steps = 3
          }.toStream: _*)(sub)
          null
        })
      }(log)
    } finally {
      registration.foreach(_.stop()(s3client, ec2client))
    }
  }

  def numSteps = transitions * 2 + 1
}


