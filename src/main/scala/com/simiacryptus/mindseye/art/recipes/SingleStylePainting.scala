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
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{BasicOptimizer, _}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, Precision}
import com.simiacryptus.mindseye.opt.Step
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object SingleStylePaintingEC2 extends SingleStylePainting with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override val styleUrl: String = "s3://simiacryptus/photos/shutterstock_87165334.jpg"
  override val contentUrl: String = "s3://simiacryptus/photos/0DSC_0005.JPG"
  override val initUrl: String = "s3://simiacryptus/photos/0DSC_0005.JPG"

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

  override def inputTimeoutSeconds = 300

  override def maxHeap: Option[String] = Option("50g")

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (15 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (15 * CudaMemory.GiB).toString
  )
}

object SingleStylePainting extends SingleStylePainting with LocalRunner[Object] with NotebookRunner[Object]

class SingleStylePainting extends ArtSetup[Object] {
  val contentUrl = "file:///C:/Users/andre/Downloads/IMG_20161019_151359076.jpg"
  //"file:///C:/Users/andre/Downloads/pictures/shutterstock_72395209.jpg"

  val styleUrl = "file:///C:/Users/andre/Downloads/pictures/Pikachu-Pokemon-Wallpapers-SWA0039152.jpg"
  val initUrl = contentUrl
  val bucket = "www.tigglegickle.com"

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> SingleStylePainting.this
      ))
    })
    log.out(log.jpg(VisionPipelineUtil.load(styleUrl, -1), "Input Style"))
    log.out(log.jpg(VisionPipelineUtil.load(contentUrl, -1), "Reference Content"))
    log.setArchiveHome(URI.create("s3://" + bucket + "/"))
    log.onComplete(() => {
      upload(log)
    }: Unit)

    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(contentUrl, initUrl, canvas, sub.eval(() => {
          new CartesianStyleNetwork(
            styleLayers = List(
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
            ),
            styleModifiers = List(
              new GramMatrixEnhancer(),
              new MomentMatcher()
            ),
            styleUrl = List(styleUrl),
            precision = Precision.Float,
            magnification = 40
          ).withContent(List(
            VGG19_0.prependAvgPool(16),
            VGG19_1b2.appendMaxPool(4),
            VGG19_1c4.appendMaxPool(2)
          ), List(
            new ContentMatcher().scale(1e2)
          ))
        }), new BasicOptimizer {
          override val trainingMinutes: Int = 60
          override val trainingIterations: Int = 50
          override val maxRate = 1e9

          override def onStepComplete(trainable: Trainable, currentPoint: Step) = {
            setPrecision(trainable, Precision.Float)
          }

          override def onStepFail(trainable: Trainable, currentPoint: Step): Boolean = {
            setPrecision(trainable, Precision.Double)
          }

          override def onComplete()(implicit log: NotebookOutput): Unit = {
            upload(log)
          }
        }, new GeometricResolutionSequence {
          override val minResolution = 800
          override val maxResolution = 800
          override val resolutionSteps = 1
        }.resolutions: _*)(sub)
        null
      })
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(contentUrl, initUrl, canvas, sub.eval(() => {
          new CartesianStyleNetwork(
            styleLayers = List(
              //              VGG19_1c1,
              //              VGG19_1c2,
              VGG19_1c3,
              VGG19_1c4,
              VGG19_1d1,
              VGG19_1d2,
              VGG19_1d3,
              VGG19_1d4
            ),
            styleModifiers = List(
              new GramMatrixEnhancer(), //.setMinMax(-0.25, 0.25),
              new MomentMatcher()
            ),
            styleUrl = List(styleUrl),
            precision = Precision.Float
          ).withContent(List(
            VGG19_0.prependAvgPool(128),
            VGG19_1c4.appendMaxPool(16),
            VGG19_1d4.appendMaxPool(8)
          ), List(
            new ContentMatcher().scale(1e0)
          ))
        }), new BasicOptimizer {
          override val trainingMinutes: Int = 180
          override val trainingIterations: Int = 20
          override val maxRate = 1e9

          override def onStepComplete(trainable: Trainable, currentPoint: Step) = {
            setPrecision(trainable, Precision.Float)
          }

          override def onStepFail(trainable: Trainable, currentPoint: Step): Boolean = {
            setPrecision(trainable, Precision.Double)
          }

          override def onComplete()(implicit log: NotebookOutput): Unit = {
            upload(log)
          }

        }, new GeometricResolutionSequence {
          override val minResolution = 1200
          override val maxResolution = 4000
          override val resolutionSteps = 3
        }.resolutions: _*)(sub)
        null
      })
    }(log)
  }
}


