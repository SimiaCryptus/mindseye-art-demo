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
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner

object SingleStylePaintingEC2 extends SingleStylePainting with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override val s3bucket = "www.tigglegickle.com"

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

  override def inputTimeoutSeconds = 30

  override def maxHeap: Option[String] = Option("50g")

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (15 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (15 * CudaMemory.GiB).toString
  )

}

object SingleStylePainting extends SingleStylePainting with LocalRunner[Object] with NotebookRunner[Object]

class SingleStylePainting extends ArtSetup[Object] {
  val contentUrl = "upload:mask"
  val styleUrl = "upload:style"
  val initUrl: String = "50 + noise * 50"
  val s3bucket: String = ""

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    implicit val _log = log
    log.setArchiveHome(URI.create(s"s3://$s3bucket/${getClass.getSimpleName.stripSuffix("$")}/${UUID.randomUUID()}/"))
    log.onComplete(() => upload(log): Unit)

    log.h1("Inputs")
    log.h2("Style")
    val styleImage = ImageArtUtil.load(log, styleUrl, 600)

    log.out(log.jpg(styleImage, "Input Style"))
    log.h2("Content")
    log.out(log.jpg(ImageArtUtil.load(log, contentUrl, 600), "Reference Content"))
    val canvas = new AtomicReference[Tensor](null)
    val registration = registerWithIndexJPG(canvas.get())

    log.h1("Painting")
    try {
      withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
        log.subreport((sub: NotebookOutput) => {
          paint(contentUrl, initUrl, canvas, sub.eval(() => {
            new VisualStyleContentNetwork(
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
              contentLayers = List(
                VGG19_1b2,
                VGG19_1c2,
                VGG19_1c4,
                VGG19_1d2,
                VGG19_1d4
              ).flatMap(baseLayer => List(
                baseLayer,
                baseLayer.prependAvgPool(2),
                baseLayer.prependAvgPool(3)
              )), contentModifiers = List(
                new ContentMatcher().scale(1e0)
              )) + new VisualStyleNetwork(
              styleLayers = List(
                VGG19_0a
              ),
              styleModifiers = List(
                new MomentMatcher()
              ).map(_.scale(1e2)),
              styleUrl = List(contentUrl),
              magnification = 1
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
        }, UUID.randomUUID().toString)
        log.subreport((sub: NotebookOutput) => {
          paint(contentUrl, initUrl, canvas, sub.eval(() => {
            new VisualStyleNetwork(
              styleLayers = List(
                VGG19_1a,
                VGG19_1b1,
                VGG19_1b2,
                VGG19_1c1,
                VGG19_1c2,
                VGG19_1c3,
                VGG19_1c4
              ).flatMap(baseLayer => List(
                baseLayer
              )),
              styleModifiers = List(
                //                new GramMatrixEnhancer(),
                new MomentMatcher()
              ),
              styleUrl = List(styleUrl),
              magnification = 1
            ) + new VisualStyleNetwork(
              styleLayers = List(
                VGG19_1c1,
                VGG19_1c2,
                VGG19_1c3,
                VGG19_1c4,
                VGG19_1d1,
                VGG19_1d2,
                VGG19_1d3,
                VGG19_1d4
              ).flatMap(baseLayer => List(
                baseLayer.prependAvgPool(2),
                baseLayer.prependAvgPool(3),
                baseLayer.prependAvgPool(4),
                baseLayer.prependAvgPool(5)
              )),
              styleModifiers = List(
                new GramMatrixEnhancer(),
                new MomentMatcher()
              ),
              styleUrl = List(styleUrl),
              magnification = 1
            ).withContent(List(
              VGG19_1b1,
              VGG19_1c2
            ).flatMap(baseLayer => List(
              baseLayer,
              baseLayer.prependAvgPool(2),
              baseLayer.prependAvgPool(3),
              baseLayer.prependAvgPool(4),
              baseLayer.prependAvgPool(5)
            )), List(
              new ContentMatcher().scale(1e0)
            ))
          }), new BasicOptimizer {
            override val trainingMinutes: Int = 90
            override val trainingIterations: Int = 20
            override val maxRate = 1e9
          }, new GeometricSequence {
            override val min: Double = 1000
            override val max: Double = 2400
            override val steps = 4
          }.toStream: _*)(sub)
          null
        }, UUID.randomUUID().toString)
      }(log)
    } finally {
      registration.foreach(_.stop()(s3client, ec2client))
    }

  }


}


