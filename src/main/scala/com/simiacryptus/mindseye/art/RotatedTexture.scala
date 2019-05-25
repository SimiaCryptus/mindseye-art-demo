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

import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.util.{ArtSetup, BasicOptimizer, CartesianStyleNetwork, Permutation}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.lang.{Coordinate, Tensor}
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.layers.cudnn.ImgBandBiasLayer
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer
import com.simiacryptus.mindseye.layers.java.{ImgViewLayer, LinearActivationLayer, SumInputsLayer}
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner._
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object RotatedTexture_EC2 extends RotatedTexture with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object RotatedTexture_Local extends RotatedTexture with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class RotatedTexture extends ArtSetup[Object] with BasicOptimizer {

  val inputUrl = "plasma"
  val tiledViewPadding = 32
  val tileSize = 400
  val tilePadding = 32
  val aspect = 1 // (11.0 - 0.5) / (8.5 - 0.5)
  val minResolution: Double = 128
  val maxResolution: Double = 512
  val resolutionSteps: Int = 4
  val rotationalSegments = 4
  val rotationalChannelPermutation = Permutation.roots(3, rotationalSegments).head.indices

  override def cudaLog = false

  def resolutions = Stream.iterate(minResolution)(_ * growth).takeWhile(_ <= maxResolution).map(_.toInt).toArray

  private def growth = Math.pow(maxResolution / minResolution, 1.0 / resolutionSteps)

  def precision(w: Int) = if (w < 200) Precision.Double else Precision.Float

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> RotatedTexture.this
      ))
    })
    for ((name, styleNetwork) <- Map(
      "cesar-domela-1" -> CartesianStyleNetwork.DOMELA_1,
      "claude-monet-1" -> CartesianStyleNetwork.MONET_1
    )) {
      log.h1(name)
      val canvas = new AtomicReference[Tensor](null)
      val progress = new AtomicReference[() => Tensor](null)
      val animationCallback = new AtomicReference[() => Seq[Tensor]](null)
      withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
        withMonitoredGif(() => Option(animationCallback.get().apply()).map(_.map(_.toRgbImage)).orNull) {
          withMonitoredJpg(() => Option(progress.get()).map(_.apply()).map(_.toRgbImage).orNull) {
            log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
              paint(
                canvas = canvas,
                rendering = progress,
                animation = animationCallback,
                network = styleNetwork
              )(sub)
              null
            })
            null
          } (log)
        } (log)
        null
      } (log)
    }
    null
  }

  def paint(canvas: AtomicReference[Tensor], rendering: AtomicReference[()=>Tensor], animation: AtomicReference[()=>Seq[Tensor]], network: CartesianStyleNetwork)(implicit sub: NotebookOutput): Unit = {
    rendering.set(()=>{
      val canvasImg = canvas.get
      if(null == canvasImg) null
      else getKaleidoscope(canvasImg.getDimensions).eval(canvasImg).getDataAndFree.getAndFree(0)
    })
    for (res <- resolutions) {
      val precision: Precision = this.precision(res)
      CudaSettings.INSTANCE().defaultPrecision = precision
      sub.h1("Resolution " + res)
      if (null == canvas.get) {
        implicit val nullNotebookOutput = new NullNotebookOutput()
        canvas.set(load(Array(res, (res * aspect).toInt, 3), inputUrl))
      }
      else {
        canvas.set(Tensor.fromRGB(TestUtil.resize(canvas.get.toRgbImage, res, true)))
      }
      val canvasDims = canvas.get.getDimensions
      val kaleidoscope = getKaleidoscope(canvasDims)
      val viewLayer = kaleidoscope.copyPipeline()
      viewLayer.wrap(new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
        .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)).freeRef()
      animation.set(()=>{
        val image = kaleidoscope.eval(canvas.get).getDataAndFree.getAndFree(0)
        val arc = 2 * Math.PI / rotationalSegments
        val permutation = Permutation(this.rotationalChannelPermutation: _*).matrix
        val identity = Permutation.unity(3).matrix
        val frames = 16
        (0.0 until 1.0 by 1.0 / frames).map(time => {
          val rotor = getRotor(arc * time, image.getDimensions)
          val sin = Math.sin(0.5 * Math.PI * time)
          val cos = Math.cos(0.5 * Math.PI * time)
          val root = permutation.scalarMultiply(sin).add(identity.scalarMultiply(cos))
          val paletteBias = new ImgBandBiasLayer(3)
          for (i <- (0 until 3)) {
            if (rotationalChannelPermutation(i) < 0) paletteBias.getBias()(i) = 256 * sin
          }
          val palette = new SimpleConvolutionLayer(1, 1, 3, 3)
          palette.kernel.setByCoord((c: Coordinate) => {
            val x = c.getCoords()(2)
            root.getEntry(x % 3, x / 3)
          })
          val frameView = PipelineNetwork.wrap(1, rotor, palette, paletteBias)
          try {
            frameView.eval(image).getDataAndFree.getAndFree(0)
          } finally {
            frameView.freeRef()
          }
        })
      })
      optimize(canvas.get, network.copy(
        precision = precision,
        viewLayer = viewLayer
      ).apply(canvas.get))
    }
  }

  def getKaleidoscope(canvasDims: Array[Int]) = {
    val permutation = Permutation(this.rotationalChannelPermutation: _*)
    require(permutation.unity == (permutation ^ rotationalSegments), s"$permutation ^ $rotationalSegments => ${(permutation ^ rotationalSegments)} != ${permutation.unity}")
    val network = new PipelineNetwork(1)
    network.add(new SumInputsLayer(), (0 until rotationalSegments)
      .map(segment => {
        if (0 == segment) network.getInput(0) else {
          network.wrap(
            getRotor(segment * 2 * Math.PI / rotationalSegments, canvasDims).setChannelSelector((permutation ^ segment).indices: _*),
            network.getInput(0)
          )
        }
      }): _*).freeRef()
    network.wrap(new LinearActivationLayer().setScale(1.0 / rotationalSegments).freeze()).freeRef()
    network
  }

  def getRotor(radians: Double, canvasDims: Array[Int]) = {
    new ImgViewLayer(canvasDims(0), canvasDims(1), true)
      .setRotationCenterX(canvasDims(0) / 2)
      .setRotationCenterY(canvasDims(1) / 2)
      .setRotationRadians(radians)
  }

}


