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
import java.util.concurrent.TimeUnit
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
import com.simiacryptus.sparkbook.util.LocalRunner

object WikiartStyle extends ArtSetup[Object] with LocalRunner[Object] with NotebookRunner[Object] {
  val contentUrl = "file:///C:/Users/andre/Downloads/pictures/P17.jpg"
  val initUrl = contentUrl
  //    "file:///H:/SimiaCryptus/data-science-tools/report/20190709080125/etc/image_17e96b4b2f8ebf.jpg"
  //    "50+noise50"
  val artist =
  "jackson-pollock"
  //  "pablo-picasso"
  //  "vladimir-tretchikoff"
  //"salvador-dali"
  //    "claude-monet"
  //    "m-c-escher"
  // "vincent-van-gogh"
  // "edvard-munch"
  val s3bucket = "www.tigglegickle.com"

  override def inputTimeoutSeconds = 5

  override def postConfigure(log: NotebookOutput) = {
    log.setArchiveHome(URI.create(s"s3://$s3bucket/${getClass.getSimpleName.stripSuffix("$")}/${UUID.randomUUID()}/"))
    log.onComplete(() => upload(log): Unit)

    log.out(log.jpg(VisionPipelineUtil.load(contentUrl, -1), "Reference Content"))
    //    val smallPaintings = "file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads6.wikiart.org/full-fathom-five(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads0.wikiart.org/easter-and-the-totem(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads2.wikiart.org/stenographic-figure(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads6.wikiart.org/the-key(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads1.wikiart.org/the-tea-cup(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads6.wikiart.org/blue-moby-dick(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads7.wikiart.org/male-and-female(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads7.wikiart.org/moon-woman-1942(1).jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/jackson-pollock/going-west.jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/jackson-pollock/the-moon-woman-cuts-the-circle-1943.jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/uploads3.wikiart.org/composition-with-pouring-ii(1).jpg".split(";")
    //    val largePaintings = "file:///H:/SimiaCryptus/data-science-tools/wikiart/paul-jackson-pollock/not-detected.jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/jackson-pollock/number-7-out-of-the-web-1949.jpg;file:///H:/SimiaCryptus/data-science-tools/wikiart/jackson-pollock/convergence-1952.jpg".split(";")
    val List(
    smallPaintings,
    largePaintings
    ) = List(
      600,
      1200
    ).map(getPaintingsByArtist(artist, _)).map(userSelect(_)(log)).map(_.get(6000, TimeUnit.SECONDS).filter(_._2).keys.toList)
    log.p("Small Images: " + smallPaintings.mkString(";"))
    log.p("Large Images: " + largePaintings.mkString(";"))

    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(contentUrl, initUrl, canvas, sub.eval(() => {
          new VisualStyleNetwork(
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
            styleUrl = smallPaintings,
            precision = Precision.Float
          ).withContent(List(
            VGG19_0b.prependAvgPool(16),
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
        }, new GeometricSequence {
          override val min: Double = 120
          override val max: Double = 800
          override val steps = 6
        }.toStream: _*)(sub)
        null
      })
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(contentUrl, initUrl, canvas, sub.eval(() => {
          new VisualStyleNetwork(
            styleLayers = List(
              VGG19_1b1,
              VGG19_1b2,
              VGG19_1c1,
              VGG19_1c2,
              VGG19_1c3,
              VGG19_1c4
            ),
            styleModifiers = List(
              new GramMatrixEnhancer(), //.setMinMax(-0.25, 0.25),
              new MomentMatcher()
            ),
            styleUrl = largePaintings,
            precision = Precision.Float
          ).withContent(List(
            VGG19_0b.prependAvgPool(128),
            VGG19_1c1.appendMaxPool(16)
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

        }, new GeometricSequence {
          override val min: Double = 1000
          override val max: Double = 2400
          override val steps = 4
        }.toStream: _*)(sub)
        null
      })
    }(log)
  }
}


