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
import com.simiacryptus.mindseye.art.util.ArtSetup.{ec2client, s3client}
import com.simiacryptus.mindseye.art.util.{BasicOptimizer, _}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory
import com.simiacryptus.mindseye.layers.java.ImgViewLayer
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object SingleStyleTextureEC2 extends SingleStyleTexture with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override val styleUrl: String = "s3://simiacryptus/photos/shutterstock_468243743.jpg"
  override val s3bucket = "www.tigglegickle.com"

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

  override def inputTimeoutSeconds = 30

  override def maxHeap: Option[String] = Option("50g")

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (15 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (15 * CudaMemory.GiB).toString
  )

}

object SingleStyleTexture extends SingleStyleTexture with LocalRunner[Object] with NotebookRunner[Object]

class SingleStyleTexture extends ArtSetup[Object] {

  val styleUrl =
    "file:///C:/Users/andre/Downloads/pictures/the-starry-night.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_240121861.jpg" // Grafiti
  //    "file:///C:/Users/andre/Downloads/pictures/1920x1080-kaufman_63748_5.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/Pikachu-Pokemon-Wallpapers-SWA0039152.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_1060865300.jpg" // Plasma Ball
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_468243743.jpg" // Leaves

  val initUrl: String = "50+noise50"
  val s3bucket: String = ""

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    log.setArchiveHome(URI.create(s"s3://$s3bucket/${getClass.getSimpleName.stripSuffix("$")}/${UUID.randomUUID()}/"))
    log.onComplete(() => upload(log): Unit)

    log.out(log.jpg(VisionPipelineUtil.load(styleUrl, 600), "Input Style"))
    val canvas = new AtomicReference[Tensor](null)
    val registration = registerWithIndex(canvas)
    try {
      withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
        log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
          paint(styleUrl, initUrl, canvas, sub.eval(() => {
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
                //                new GramMatrixEnhancer(),
                new MomentMatcher()
              ),
              styleUrl = List(styleUrl),
              magnification = 2,
              viewLayer = dims => {
                val padding = Math.max(16, dims(0) / 4)
                new ImgViewLayer(dims(0) + padding, dims(1) + padding, true)
                  .setOffsetX(-padding / 2).setOffsetY(-padding / 2)
              }
            )
          }), new BasicOptimizer {
            override val trainingMinutes: Int = 60
            override val trainingIterations: Int = 50
            override val maxRate = 1e9
          }, new GeometricSequence {
            override val min: Double = 320
            override val max: Double = 800
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


