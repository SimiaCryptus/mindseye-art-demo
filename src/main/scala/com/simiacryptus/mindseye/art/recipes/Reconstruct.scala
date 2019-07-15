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

import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{BasicOptimizer, _}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.opt.Step
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object Reconstruct extends ArtSetup[Object] with LocalRunner[Object] with NotebookRunner[Object] {
  val original = "file:///C:/Users/andre/Downloads/img11262015_0645.jpg"
  val initUrl =
    "50+noise50"
  //    original
  val bucket = "www.tigglegickle.com"
  val reference = "file:///C:/Users/andre/Downloads/img11262015_0645.jpg".split("\n")

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> Reconstruct.this
      ))
    })
    log.out(log.jpg(VisionPipelineUtil.load(original, -1), "Reference Content"))
    log.setArchiveHome(URI.create("s3://" + bucket + "/"))
    log.onComplete(() => {
      upload(log)
    }: Unit)

    log.p("Reference Images: " + reference.mkString(";"))

    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(original, initUrl, canvas, sub.eval(() => {
          new CartesianStyleNetwork(
            styleLayers = List(
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
              new MomentMatcher(),
              new GramMatrixEnhancer() //.setMinMax(-.25,.25)
            ),
            styleUrl = reference,
            precision = Precision.Float
          ).withContent(List(
            VGG19_0.prependAvgPool(8)
          ), List(
            new ContentMatcher().scale(1e1)
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
          override val minResolution = 400 // referenceImage.getWidth / 4
          override val maxResolution = 1200 // referenceImage.getWidth
          override val resolutionSteps = 5
        }.resolutions: _*)(sub)
        null
      })
    }(log)
  }
}


