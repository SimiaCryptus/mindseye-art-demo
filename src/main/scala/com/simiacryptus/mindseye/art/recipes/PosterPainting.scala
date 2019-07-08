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

import java.io.File
import java.util
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.mindseye.art.models.PoolingPipeline._
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.models.VGG19._
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.{BasicOptimizer, _}
import com.simiacryptus.mindseye.art.{SumTrainable, TiledTrainable}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, MultiPrecision, Precision}
import com.simiacryptus.mindseye.network.DAGNetwork
import com.simiacryptus.mindseye.opt.Step
import com.simiacryptus.notebook.{FormQuery, MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

import scala.collection.mutable

object PosterPainting extends ArtSetup[Object] with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5

  val contentUrl = "file:///C:/Users/andre/Downloads/IMG_20190609_154147991.jpg"
  //  "file:///C:/Users/andre/Downloads/IMG_20190528_113653502.jpg"

  def initUrl = "50+noise50"

  val artist = "pablo-picasso"
  //"vladimir-tretchikoff"
  //"salvador-dali"
  //    "claude-monet"
  //    "m-c-escher"
  // "vincent-van-gogh"
  // "edvard-munch"

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> PosterPainting.this
      ))
    })
    val canvas = new AtomicReference[Tensor](null)
    val smallPaintings = userSelect(getPaintingsByArtist(artist, 600))(log)
    val largePaintings = userSelect(getPaintingsByArtist(artist, 1400))(log)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, new CartesianStyleNetwork(
          styleLayers = List(
            VGG19.VGG19_1b1,
            VGG19.VGG19_1b2,
            VGG19.VGG19_1c1,
            VGG19.VGG19_1c2,
            VGG19.VGG19_1c3,
            VGG19.VGG19_1c4,
            VGG19.VGG19_1d1,
            VGG19.VGG19_1d2,
            VGG19.VGG19_1d3,
            VGG19.VGG19_1d4
          ),
          styleModifiers = List(
            new GramMatrixEnhancer().scale(1.0),
            new MomentMatcher().setTileSize(400)
          ),
          styleUrl = smallPaintings,
          magnification = 1.0,
          maxWidth = 4000,
          maxPixels = 5e7,
          tileSize = 1200,
          tilePadding = 0,
          precision = Precision.Double
        ).withContent(List(
          Pooling32,
          VGG19.VGG19_1d1,
          VGG19.VGG19_1d2,
          VGG19.VGG19_1d3,
          VGG19.VGG19_1d4
        ), List(
          new ContentMatcher().scale(1e0)
        )), new BasicOptimizer {
          override val trainingMinutes: Int = 90
          override val trainingIterations: Int = 50
          override val maxRate = 1e9

          override def onStepComplete(trainable: Trainable, currentPoint: Step) = {
            if (currentPoint.iteration > 3) {
              CudaSettings.INSTANCE().defaultPrecision = Precision.Float
              setPrecision(trainable, Precision.Float)
            }
          }
        }, new GeometricResolutionSequence {
          override val minResolution = 180
          override val maxResolution = 800
          override val resolutionSteps = 5
        }.resolutions: _*)(sub)
        null
      })
      null
    }(log)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, new CartesianStyleNetwork(
          styleLayers = List(
            VGG19.VGG19_1b1,
            VGG19.VGG19_1b2,
            VGG19.VGG19_1c1,
            VGG19.VGG19_1c2,
            VGG19.VGG19_1c3,
            VGG19.VGG19_1c4
          ),
          styleModifiers = List(
            new GramMatrixEnhancer().setMinMax(-0.5, 0.5).scale(1e0),
            new MomentMatcher()
          ),
          styleUrl = largePaintings,
          magnification = 1.0,
          maxWidth = 14000,
          maxPixels = 5e8,
          tileSize = 1400,
          tilePadding = 128,
          precision = Precision.Double
        ).withContent(List(
          Pooling128,
          VGG19_1c1
        ), List(
          new ContentMatcher().scale(1e0)
        )), new BasicOptimizer {
          override val trainingMinutes: Int = 90
          override val trainingIterations: Int = 20
          override val maxRate = 1e9

          override def onStepComplete(trainable: Trainable, currentPoint: Step) = {
            if (currentPoint.iteration > 3) {
              CudaSettings.INSTANCE().defaultPrecision = Precision.Float
              setPrecision(trainable, Precision.Float)
            }
          }
        }, new GeometricResolutionSequence {
          override val minResolution = 1024
          override val maxResolution = 1800
          override val resolutionSteps = 5
        }.resolutions: _*)(sub)
        null
      })
      null
    }(log)

    null
  }

  private val pngCache = new mutable.HashMap[String, File]()
  def userSelect(paintings: Array[String])(implicit log: NotebookOutput): List[String] = {
    val ids = paintings.map(_ -> UUID.randomUUID().toString).toMap
    new FormQuery[Map[String, Boolean]](log.asInstanceOf[MarkdownNotebookOutput]) {

      override def height(): Int = 800

      override protected def getFormInnerHtml: String = {
        (for ((url, v) <- getValue.filter(_._2)) yield {
          val filename = url.split('/').last.toLowerCase.stripSuffix(".png") + ".png"
          pngCache.getOrElseUpdate(url, log.pngFile(VisionPipelineUtil.load(url, 256), new File(log.getResourceDir, filename)))
          s"""<input type="checkbox" name="${ids(url)}" value="true"><img src="etc/$filename"><br/>"""
        }).mkString("\n")
      }

      override def valueFromParams(parms: util.Map[String, String]): Map[String, Boolean] = {
        (for ((k, v) <- getValue) yield {
          k -> parms.getOrDefault(ids(k), "false").toBoolean
        })
      }
    }.setValue(paintings.map(_ -> true).toMap).print().get(6000, TimeUnit.SECONDS).filter(_._2).keys.toList
  }

  private def setPrecision(trainable: Trainable, precision: Precision): Unit = {
    trainable match {
      case trainable: SumTrainable =>
        for (layer <- trainable.getInner) setPrecision(layer, precision)
      case trainable: TiledTrainable =>
        trainable.setPrecision(precision)
      case trainable: Trainable if null != trainable.getLayer =>
        MultiPrecision.setPrecision(trainable.getLayer.asInstanceOf[DAGNetwork], precision)
    }
  }
}


