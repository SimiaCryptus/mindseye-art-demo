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

package com.simiacryptus.mindseye.art

import java.awt.image.BufferedImage
import java.awt.{Font, GraphicsEnvironment}

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.lang.{Coordinate, Tensor}
import com.simiacryptus.mindseye.layers.java._
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner._
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

object Stereogram_EC2 extends Stereogram with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object Stereogram_Local extends Stereogram with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class Stereogram extends ArtSetup[Object] with BasicOptimizer {

  val seed = "noise200"
  val posterWidth = 1400
  val resolutions: Array[Int] = Array(50, 100)
  val aspectRatio = 14.0
  val tiledViewPadding = 32
  val depthFactor = 8
  val text = "DANTE"

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> Stereogram.this
      ))
    })

    if (false) log.subreport("Fonts", (subreport: NotebookOutput) => Future {
      GraphicsEnvironment.getLocalGraphicsEnvironment.getAvailableFontFamilyNames.filter((x: String) => !(x == "EmojiOne Color")).foreach((fontname: String) => {
        subreport.p(fontname)
        subreport.p(subreport.png(TextUtil.draw(text, 800, 20, fontname, Font.PLAIN), fontname))
        subreport.p(subreport.png(TextUtil.draw(text, 800, 20, fontname, Font.ITALIC), fontname))
        subreport.p(subreport.png(TextUtil.draw(text, 800, 20, fontname, Font.BOLD), fontname))
      })
      null
    })

    for ((name, styleNetwork) <- Map(
      "cesar-domela-1" -> CartesianStyleNetwork.DOMELA_1,
      "claude-monet-1" -> CartesianStyleNetwork.MONET_1
    )) {
      var canvas: Tensor = null
      lazy val depthImg = depthMap(text)
      withMonitoredJpg(() => stereoImage(depthMap = depthImg, canvas = canvas, depthFactor)) {
        log.subreport(name, (sub: NotebookOutput) => {
          for (res <- resolutions) {
            val precision: Precision = this.precision(res)
            CudaSettings.INSTANCE().defaultPrecision = precision
            sub.h1("Resolution " + res)
            if (null == canvas) {
              canvas = load(Array(res, (res * aspectRatio).toInt, 3), seed)(new NullNotebookOutput())
            }
            else {
              canvas = Tensor.fromRGB(TestUtil.resize(canvas.toRgbImage, res, true))
            }
            val canvasDims = canvas.getDimensions
            val viewLayer = new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
              .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)
            optimize(canvas, styleNetwork.copy(precision = precision, viewLayer = viewLayer).apply(canvas))(sub)
          }
          null
        })
        null
      } {
        log
      }
    }
    null
  }

  def precision(width: Int) = if (width < 0) Precision.Double else Precision.Float

  def depthMap(text: String, fontName: String = "Calibri"): Tensor =
    Tensor.fromRGB(TextUtil.draw(text, posterWidth, 120, fontName, Font.BOLD | Font.CENTER_BASELINE))

  def stereoImage(depthMap: Tensor, canvas: Tensor, depthFactor: Int): BufferedImage = {
    val dimensions = canvas.getDimensions
    val canvasWidth = dimensions(0)
    val depthScale = canvasWidth / (depthFactor * depthMap.getData.max)

    def getPixel(x: Int, y: Int, c: Int): Double = {
      if (x < 0) getPixel(x + canvasWidth, y, c)
      else if (x < canvasWidth) canvas.get(x, y % dimensions(1), c)
      else {
        val depth = depthMap.get(x, y, c)
        if (0 == depth) canvas.get(x % canvasWidth, y % dimensions(1), c)
        else getPixel(x - canvasWidth + (depthScale * depth).toInt, y, c)
      }
    }

    depthMap.copy().setByCoord((c: Coordinate) => {
      val ints = c.getCoords()
      getPixel(ints(0), ints(1), ints(2))
    }, true).toRgbImage
  }


}

