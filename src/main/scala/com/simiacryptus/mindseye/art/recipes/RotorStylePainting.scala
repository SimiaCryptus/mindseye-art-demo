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
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object RotorStylePaintingEC2 extends RotorStylePainting with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override val styleUrl: String = "s3://simiacryptus/photos/vangogh-starry-night-ballance1.jpg"
  override val contentUrl: String = "s3://simiacryptus/photos/40fe5e10-1448-45f8-89ae-90ffacb30b7d.jpeg"
  override val s3bucket = "www.tigglegickle.com"

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

  override def inputTimeoutSeconds = 30

  override def maxHeap: Option[String] = Option("50g")

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (15 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (15 * CudaMemory.GiB).toString
  )

}

object RotorStylePainting extends RotorStylePainting with LocalRunner[Object] with NotebookRunner[Object]

class RotorStylePainting extends RotorArt {
  //  override lazy val rotationalChannelPermutation: Array[Int] = Array(-3, -1, -2)
  override val rotationalChannelPermutation: Array[Int] = Array(1, 2, 3)
  override val rotationalSegments: Int = 6
  val contentUrl =
    "file:///C:/Users/andre/Downloads/pictures/40fe5e10-1448-45f8-89ae-90ffacb30b7d.jpeg"
  //    "file:///C:/Users/andre/Downloads/pictures/Pikachu-Pokemon-Wallpapers-SWA0039152.jpg"
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_1060865300.jpg" // Plasma Ball
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_468243743.jpg" // Leaves
  //    "file:///C:/Users/andre/Downloads/pictures/shutterstock_248374732_centered.jpg" // Earth
  val styleUrl =
  //    "file:///C:/Users/andre/Downloads/pictures/the-starry-night.jpg"
  "file:///C:/Users/andre/Downloads/pictures/shutterstock_240121861.jpg" // Grafiti
  //  "file:///C:/Users/andre/Downloads/pictures/1920x1080-kaufman_63748_5.jpg"
  val initUrl: String = "50+noise50"
  val s3bucket: String = "www.tigglegickle.com"

  override def inputTimeoutSeconds = 30

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    log.setArchiveHome(URI.create(s"s3://$s3bucket/${getClass.getSimpleName.stripSuffix("$")}/${UUID.randomUUID()}/"))
    log.onComplete(() => upload(log): Unit)

    log.out(log.jpg(ImageArtUtil.load(log, styleUrl, 600), "Input Style"))
    log.out(log.jpg(ImageArtUtil.load(log, contentUrl, 600), "Reference Content"))
    val canvas = new AtomicReference[Tensor](null)
    val renderingFn: Seq[Int] => PipelineNetwork = dims => {
      getKaleidoscope(dims.toArray).copyPipeline()
    }
    val registration = registerWithIndexJPG({
      val tensor = canvas.get()
      renderingFn(tensor.getDimensions).eval(tensor).getDataAndFree.getAndFree(0)
    })
    try {
      withMonitoredJpg(() => Option(canvas.get()).map(tensor => {
        renderingFn(tensor.getDimensions).eval(tensor).getDataAndFree.getAndFree(0).toRgbImage
      }).orNull) {
        log.subreport("Painting", (sub: NotebookOutput) => {
          paint(
            contentUrl = contentUrl,
            initFn = (contentTensor: Tensor) => ArtUtil.load(contentTensor, initUrl),
            canvas = canvas,
            network = sub.eval(() => {
              new VisualStyleContentNetwork(
                styleLayers = List(
                  //                VGG19_1a,
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
                  baseLayer.prependAvgPool(2)
                  //                baseLayer.prependAvgPool(3)
                )
                ),
                styleModifiers = List(
                  new GramMatrixEnhancer(),
                  new MomentMatcher()
                ),
                styleUrl = List(styleUrl),
                magnification = 8,
                contentLayers = List(
                  //                VGG19_1b2,
                  VGG19_1c2,
                  VGG19_1c4,
                  VGG19_1d2
                  //                VGG19_1d4
                ).flatMap(baseLayer => List(
                  baseLayer,
                  baseLayer.prependAvgPool(2)
                  //                baseLayer.prependAvgPool(3)
                )
                ).flatMap(baseLayer => List(
                  baseLayer.appendMaxPool(2)
                )
                ), contentModifiers = List(
                  new ContentMatcher().scale(5e0)
                ),
                viewLayer = renderingFn
                //            ) + new VisualStyleNetwork(
                //              styleLayers = List(
                //                VGG19_0a
                //              ),
                //              styleModifiers = List(
                //                new MomentMatcher()
                //              ).map(_.scale(1e0)),
                //              styleUrl = List(contentUrl),
                //              magnification = 1,
                //              viewLayer = renderingFn
              )
            }), optimizer = new BasicOptimizer {
              override val trainingMinutes: Int = 90
              override val trainingIterations: Int = 30
              override val maxRate = 1e9

              override def renderingNetwork(dims: Seq[Int]): PipelineNetwork = renderingFn(dims)
            }, renderingFn = renderingFn, resolutions = new GeometricSequence {
              override val min: Double = 200
              override val max: Double = 1800
              override val steps = 5
            }.toStream)(sub)
          null
        })
      }(log)
    } finally {
      registration.foreach(_.stop()(s3client, ec2client))
    }

  }


}


