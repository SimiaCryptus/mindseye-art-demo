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

import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.mindseye.art.models.PoolingPipeline._
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.{BasicOptimizer, _}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object PosterPainting extends ArtSetup[Object] with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5

  val contentUrl = "file:///C:/Users/andre/Downloads/IMG_20190528_113653502.jpg"
  def initUrl = "50+noise50"
  val artist = "edvard-munch"

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> PosterPainting.this
      ))
    })
    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, new CartesianStyleNetwork(
          styleLayers = List(
            VGG19.VGG19_1a,
            VGG19.VGG19_1b1,
            VGG19.VGG19_1c1,
            VGG19.VGG19_1c2,
            VGG19.VGG19_1c3,
            VGG19.VGG19_1c4
          ),
          styleModifiers = List(
            new GramMatrixEnhancer().setMinMax(-0.5,0.5).scale(0.5),
            new MomentMatcher()
          ),
          styleUrl = getPaintingsByArtist(artist, 400).take(50),
          magnification = 1.0,
          maxWidth = 4000,
          maxPixels = 5e7,
          tileSize = 400,
          tilePadding = 8,
          precision = Precision.Double
        ).withContent(List(
          Pooling8,
          VGG19.VGG19_1b1
        ), List(
          new ContentMatcher().scale(5e0)
        )), new BasicOptimizer {
          override val trainingMinutes: Int = 90
          override val trainingIterations: Int = 50
          override val maxRate = 1e9
        }, new GeometricResolutionSequence {
          override val minResolution = 180
          override val maxResolution = 320
          override val resolutionSteps = 3
        }.resolutions: _*)(sub)
        null
      })
      null
    }(log)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, new CartesianStyleNetwork(
          styleLayers = List(
            VGG19.VGG19_1a,
            VGG19.VGG19_1b2,
            VGG19.VGG19_1c1,
            VGG19.VGG19_1c3,
            VGG19.VGG19_1c4,
            VGG19.VGG19_1d1,
            VGG19.VGG19_1d4
          ),
          styleModifiers = List(
            new MomentMatcher()
          ),
          styleUrl = getPaintingsByArtist(artist, 2000).take(5),
            //getPaintingsBySearch("starry night", 2000).take(2),
          magnification = 1.0,
          maxWidth = 4000,
          maxPixels = 5e7,
          tileSize = 400,
          tilePadding = 8,
          precision = Precision.Float
        ).withContent(List(
          Pooling128,
          VGG19_1c3,
          VGG19_1c4
        ), List(
          new ContentMatcher().scale(1e0)
        )), new BasicOptimizer {
          override val trainingMinutes: Int = 90
          override val trainingIterations: Int = 20
          override val maxRate = 1e9
        }, new GeometricResolutionSequence {
          override val minResolution = 512
          override val maxResolution = 2048
          override val resolutionSteps = 5
        }.resolutions: _*)(sub)
        null
      })
      null
    }(log)

    null
  }

}


