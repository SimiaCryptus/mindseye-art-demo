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
import com.simiacryptus.mindseye.art.models.VGG19.{VGG19_1e1, _}
import com.simiacryptus.mindseye.art.ops.{ChannelMeanMatcher, GramMatrixEnhancer, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, ArtUtil, BasicOptimizer, CartesianStyleNetwork}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.layers.java.ImgViewLayer
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object TextureStyleDemo_EC2 extends TextureStyleDemo with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object TextureStyleDemo_Local extends TextureStyleDemo with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class TextureStyleDemo extends ArtSetup[Object] with BasicOptimizer {

  val inputUrl = "plasma"
  val tiledViewPadding = 32
  val tileSize = 400
  val tilePadding = 32
  val aspect = 1 // (11.0 - 0.5) / (8.5 - 0.5)
  val minResolution: Double = 128
  val maxResolution: Double = 512
  val resolutionSteps: Int = 4

  override def cudaLog = false

  def resolutions = Stream.iterate(minResolution)(_ * growth).takeWhile(_ <= maxResolution).map(_.toInt).toArray

  private def growth = Math.pow(maxResolution / minResolution, 1.0 / resolutionSteps)

  def precision(w: Int) = if (w < 200) Precision.Double else Precision.Float

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> TextureStyleDemo.this
      ))
    })
    val styles = this.styles
    for ((name, styleNetwork) <- styles) {
      log.h1(name)
      for (layers <- List(
        List(
//          VGG19_1d1,
//          VGG19_1d2,
//          VGG19_1d3,
//          VGG19_1d4
//        ), List(
          VGG19_1b2,
          VGG19_1c4,
          VGG19_1d4,
          VGG19_1e4
        ), List(
          VGG19_1a,
          VGG19_1b1,
          VGG19_1c1,
          VGG19_1d1,
          VGG19_1e1
        ), List(
          VGG19_0,
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
          VGG19_1d4,
          VGG19_1e1,
          VGG19_1e2,
          VGG19_1e3,
          VGG19_1e4
        )
      )) {
        log.h2(layers.map(_.name()).mkString(", "))
        val canvas = new AtomicReference[Tensor](null)
        withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
          log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
            paint(canvas, styleNetwork.copy(styleLayers = layers))(sub)
            null
          })
          null
        }(log)
      }
    }
    null
  }

  def styles: Map[String, CartesianStyleNetwork] = {
    Map(
      "landscsapes" -> new CartesianStyleNetwork(
        styleLayers = List(
          VGG19_1c1,
          VGG19_1e1,
          VGG19_1e3
        ),
        styleModifiers = List(
          new ChannelMeanMatcher(),
          new GramMatrixMatcher(),
          new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(400)
        ),
        styleUrl = ArtUtil.findFiles(
          """transvaal-evening-nelspruit.jpg!Large.jpg
            |seine-and-eiffel-tower-in-the-sunset-1910.jpg!Large.jpg
            |view-of-the-palace-embankment-from-st-peter-s-and-st-paul-s-fortress-1810.jpg!Large.jpg
            |landscape-with-cattle-at-limousin-1837.jpg!Large.jpg
            |figure-in-landscape-1957.jpg!Large.jpg
            |at-the-field-1890.jpg!Large.jpg
            |mount-mckinley.jpg!Large.jpg
            |plums-waterglass-and-peaches-1889.jpg!Large.jpg
            |yacht-race-1895.jpg!Large.jpg
            |sheep-in-a-landscape.jpg!Large.jpg
            |landscape-with-river-1954.jpg!Large.jpg
            |espa-o-dourado.jpg!Large.jpg
            |chiswick-house-middlesex-1742.jpg!Large.jpg
            |mountain-hike-trip-to-duke-stand.jpg!Large.jpg
            |stemship-in-sunset.jpg!Large.jpg
            |mexico-1925.jpg!Large.jpg
            |sailboats-and-fishing-boats-on-a-choppy-lake.jpg!Large.jpg
            |the-golden-tower-1833.jpg!Large.jpg
            |forest-glade-glade-1897.jpg!Large.jpg
            |by-the-sea.jpg!Large.jpg
            |bombay-1731.jpg!Large.jpg
            |italian-landscape-1847.jpg!Large.jpg
            |landscape-view-in-cumberland-1820.jpg!Large.jpg
            |life-on-the-coast-of-praslin-seychelles-1883.jpg!Large.jpg
            |the-trapper-1914.jpg!Large.jpg
            |landscape-with-figures-1757.jpg!Large.jpg
            |a-view-of-the-thames-looking-towards-battersea.jpg!Large.jpg
            |little-girl-n-a-lane-giverny-1907.jpg!Large.jpg
            |exploding-ship-1900.jpg!Large.jpg
            |gutt-paa-hvit-hest.jpg!Large.jpg
            |landscape-1874.jpg!Large.jpg
            |militia-in-the-dunes-in-ambush.jpg!Large.jpg
            |pine-grove-in-the-swamp-1873.jpg!Large.jpg
            |the-river-oka-1918.jpg!Large.jpg
            |the-free-trader-1925.jpg!Large.jpg
            |quiet-cloister-1890-3.jpg!Large.jpg
            |houses-of-parliament.jpg!Large.jpg
            |evening-at-the-sea-1941.jpg!Large.jpg
            |landscape-with-nymphs-and-satyrs-1623.jpg!Large.jpg
            |forest-landscape-1840.jpg!Large.jpg
            |not_detected_212150.jpg!Large.jpg
            |ercildoune-near-ballarat-1888.jpg!Large.jpg
            |indonesian-landscape-7.jpg!Large.jpg
            |supreme-court-1880.jpg!Large.jpg
            |morning-in-the-oasis-of-alkantra-1913.jpg!Large.jpg
            |has-been-in-desert-1909.jpg!Large.jpg
            |morning-1873-1.jpg!Large.jpg
            |the-cowherd.jpg!Large.jpg
            |banks-of-the-seine-in-summer-tournedos-sur-seine-1899.jpg!Large.jpg
            |long-shadows-cattle-on-the-island-of-saltholm-1890.jpg!Large.jpg
            |cloudy-1975.jpg!Large.jpg
            |shipwreck-1854.jpg!Large.jpg
            |winter-landscape-with-violet-lights.jpg!Large.jpg
            |study-of-sunlight.jpg!Large.jpg
            |outward-voyage-of-the-bucintoro-to-san-nicol-del-lido-1788.jpg!Large.jpg
            |the-tomb-of-george-washington.jpg!Large.jpg
            |antibes-in-the-morning(1).jpg!Large.jpg
            |the-shepherd-1905.jpg!Large.jpg
            |harvest-scene-valley-of-the-delaware-1868.jpg!Large.jpg
            |stormy-sea-with-sailing-vessels-1668.jpg!Large.jpg
            |the-dam-loing-canal-at-saint-mammes-1884.jpg!Large.jpg
            |landscape-looking-towards-sellers-hall-from-mill-bank-1818.jpg!Large.jpg
            |sky-august-1986.jpg!Large.jpg
            |cursed-field-the-place-of-execution-in-ancient-rome-crucified-slave-1878.jpg!Large.jpg""".stripMargin.split('\n').toSet
        )
      ),
      "drawings" -> new CartesianStyleNetwork(
        styleLayers = List(
          VGG19_1c1,
          VGG19_1e1,
          VGG19_1e3
        ),
        styleModifiers = List(
          new ChannelMeanMatcher(),
          new GramMatrixMatcher(),
          new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(400)
        ),
        styleUrl = ArtUtil.findFiles(
          """preliminaries-the-alpha-the-maiestas-domini-and-the-portraits-of-the-authors.jpg!Large.jpg
            |mouvement-1989.jpg!Large.jpg
            |hartley-ginny-1970.jpg!Large.jpg
            |not_detected_233129.jpg!Large.jpg
            |illustration-from-the-twelve-hours-of-the-green-houses-c-1795-colour-woodblock-print.jpg!Large.jpg
            |the-kabuki-actors-ichikawa-danjuro-vii-as-iwafuji-1824.jpg!Large.jpg
            |metamorphosis-iii-1968-2.jpg!Large.jpg
            |plum-1930.jpg!Large.jpg
            |birch-trees-1911.jpg!Large.jpg
            |d-landscape-1932.jpg!Large.jpg
            |royal-flush-1977.jpg!Large.jpg
            |pupppet-theatre-1907.jpg!Large.jpg
            |a-barbet-himalayan-blue-throated-bird-1615.jpg!Large.jpg
            |untitled-1899.jpg!Large.jpg
            |walk-of-louis-xv-in-childhood.jpg!Large.jpg
            |negro-attacked-by-a-jaguar-1910.jpg!Large.jpg
            |design-for-tulip-and-willow-indigo-discharge-wood-block-printed-fabric-1873.jpg!Large.jpg
            |job-1896.jpg!Large.jpg
            |spring-motif-view-from-stone-island-to-krestovsky-and-yelagin-islands-1904.jpg!Large.jpg
            |girl-with-a-rose-in-her-lap-1960.jpg!Large.jpg
            |untitled-1949.jpg!Large.jpg
            |the-dragon.jpg!Large.jpg
            |king-and-his-subjects-2005.jpg!Large.jpg""".stripMargin.split('\n').toSet
        )
      ),
      "abstract" -> new CartesianStyleNetwork(
        styleLayers = List(
          VGG19_1c1,
          VGG19_1d1
        ),
        styleModifiers = List(
          new ChannelMeanMatcher(),
          new GramMatrixMatcher(),
          new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(400)
        ),
        styleUrl = ArtUtil.findFiles(
          """tall-portuguese-woman-1916.jpg!Large.jpg
            |angel-of-the-last-judgment-1911.jpg!Large.jpg
            |music-1904.jpg!Large.jpg
            |wedding-ornaments-2005.jpg!Large.jpg
            |a-young-man-breaking-into-the-girls-dance-and-the-old-women-are-in-panic.jpg!Large.jpg
            |harlequin-and-clown-with-mask-1942.jpg!Large.jpg
            |the-street-enters-the-house-1911-1.jpg!Large.jpg
            |sc-ne-de-cirque.jpg!Large.jpg
            |conversation-two-nudes-in-an-interior-1978.jpg!Large.jpg
            |abstract-composition-1955.jpg!Large.jpg
            |the-two-models-1930.jpg!Large.jpg
            |composition-no-62-1917.jpg!Large.jpg
            |nebozvon-skybell-1919.jpg!Large.jpg
            |portrait-of-walter-lippman.jpg!Large.jpg
            |goya-s-lover-1977.jpg!Large.jpg
            |the-muses.jpg!Large.jpg
            |at-olympia-s-design-for-tales-of-hoffmann-by-j-offenbach-1915.jpg!Large.jpg
            |thalys-2009.jpg!Large.jpg
            |royal-flush-1977.jpg!Large.jpg
            |duel-1912.jpg!Large.jpg""".stripMargin.split('\n').toSet
        )
      )
    )
  }

  def paint(canvas: AtomicReference[Tensor], network: CartesianStyleNetwork)(implicit sub: NotebookOutput): Unit = {
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
      val viewLayer = new ImgViewLayer(canvasDims(0) + tiledViewPadding, canvasDims(1) + tiledViewPadding, true)
        .setOffsetX(-tiledViewPadding / 2).setOffsetY(-tiledViewPadding / 2)
      optimize(canvas.get, network.copy(precision = precision, viewLayer = viewLayer).apply(canvas.get))
    }
  }

}


