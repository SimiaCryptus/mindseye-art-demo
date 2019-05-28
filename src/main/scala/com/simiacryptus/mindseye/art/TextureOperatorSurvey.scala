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
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.ops._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util.{ArtSetup, BasicOptimizer, CartesianStyleNetwork, VisionPipelineUtil}
import com.simiacryptus.mindseye.eval.Trainable
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.lang.{Layer, Tensor}
import com.simiacryptus.mindseye.layers.java.ImgViewLayer
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object TextureOperatorSurvey_EC2 extends TextureOperatorSurvey with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object TextureOperatorSurvey_Local extends TextureOperatorSurvey with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class TextureOperatorSurvey extends ArtSetup[Object] with BasicOptimizer {

  val inputUrl = "noise150"
  val tiledViewPadding = 0
  val tileSize = 512
  val tilePadding = 32
  val aspect = 1 // (11.0 - 0.5) / (8.5 - 0.5)
  val minResolution: Double = 256
  val maxResolution: Double = minResolution
  val resolutionSteps: Int = 1
  val styleMagnification = 1.0
  val styleMin = 64
  val styleMax = 1280
  val stylePixelMax = 1e7

  override def trainingIterations: Int = 25

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> TextureOperatorSurvey.this
      ))
    })
    val styles = Map(
      "drawings" -> List(
        "s3a://data-cb03c/crawl/wikiart/images/abraham-manievich/birch-trees-1911.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/aladar-korosfoi-kriesch/pupppet-theatre-1907.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/alice-neel/hartley-ginny-1970.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/alphonse-mucha/job-1896.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/anna-ostroumova-lebedeva/spring-motif-view-from-stone-island-to-krestovsky-and-yelagin-islands-1904.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/audrey-flack/royal-flush-1977.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/aurel-cojan/mouvement-1989.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ende/preliminaries-the-alpha-the-maiestas-domini-and-the-portraits-of-the-authors.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ende/the-woman-garbed-by-the-sun-and-the-dragon.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/francois-arnal/untitled-1949.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/henri-rousseau/negro-attacked-by-a-jaguar-1910.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/john-bratby/girl-with-a-rose-in-her-lap-1960.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/karl-bryullov/walk-of-louis-xv-in-childhood.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/kitagawa-utamaro/illustration-from-the-twelve-hours-of-the-green-houses-c-1795-colour-woodblock-print.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/kitagawa-utamaro/not_detected_233129.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/kitagawa-utamaro/the-hour-of-the-dragon.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/m-c-escher/metamorphosis-iii-1968-2.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/otto-eckmann/untitled-1899.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/twins-seven-seven/king-and-his-subjects-2005.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/unichi-hiratsuka/plum-1930.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ustad-mansur/a-barbet-himalayan-blue-throated-bird-1615.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/utagawa-kunisada-ii/the-dragon.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/utagawa-toyokuni-ii/the-kabuki-actors-ichikawa-danjuro-vii-as-iwafuji-1824.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/william-morris/design-for-tulip-and-willow-indigo-discharge-wood-block-printed-fabric-1873.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/wladyslaw-strzeminski/d-landscape-1932.jpg!Large.jpg"
      ),
      "abstract" -> List(
        "s3a://data-cb03c/crawl/wikiart/images/albert-bloch/duel-1912.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/andre-derain/music-1904.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/andre-lanskoy/abstract-composition-1955.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/andrei-ryabushkin/a-young-man-breaking-into-the-girls-dance-and-the-old-women-are-in-panic.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/aristarkh-lentulov/nebozvon-skybell-1919.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/audrey-flack/royal-flush-1977.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/corneille/conversation-two-nudes-in-an-interior-1978.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/emilio-grau-sala/sc-ne-de-cirque.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/george-stefanescu-ramnic/wedding-ornaments-2005.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/jacoba-van-heemskerck/composition-no-62-1917.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/jonone/thalys-2009.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/max-weber/the-muses.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/r-b-kitaj/portrait-of-walter-lippman.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/rafael-zabaleta/harlequin-and-clown-with-mask-1942.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/raoul-dufy/the-two-models-1930.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/robert-delaunay/tall-portuguese-woman-1916.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/serge-sudeikin/at-olympia-s-design-for-tales-of-hoffmann-by-j-offenbach-1915.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/umberto-boccioni/the-street-enters-the-house-1911-1.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/walasse-ting/goya-s-lover-1977.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/wassily-kandinsky/angel-of-the-last-judgment-1911.jpg!Large.jpg"
      ),
      "landscsapes" -> List(
        "s3a://data-cb03c/crawl/wikiart/images/abdullah-suriosubroto/indonesian-landscape-7.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/alfred-sisley/the-dam-loing-canal-at-saint-mammes-1884.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/antonio-jacobsen/yacht-race-1895.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/arkhip-kuindzhi/sunset-in-the-steppes-by-the-sea.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/carl-spitzweg/mountain-hike-trip-to-duke-stand.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/charles-atamian/children-by-the-sea.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/charles-francois-daubigny/by-the-sea.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/charles-m-russell/mexico-1925.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/charles-m-russell/the-free-trader-1925.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/charles-martin-powell/sailboats-and-fishing-boats-on-a-choppy-lake.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/charles-willson-peale/landscape-looking-towards-sellers-hall-from-mill-bank-1818.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/claude-monet/antibes-in-the-morning(1).jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/claude-monet/houses-of-parliament.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/claude-monet/not_detected_212150.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/constantin-flondor/sky-august-1986.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/david-davies/ercildoune-near-ballarat-1888.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/david-roberts/the-golden-tower-1833.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/elmer-bischoff/figure-in-landscape-1957.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/esaias-van-de-velde/militia-in-the-dunes-in-ambush.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/francesco-guardi/outward-voyage-of-the-bucintoro-to-san-nicol-del-lido-1788.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/fyodor-alekseyev/view-of-the-palace-embankment-from-st-peter-s-and-st-paul-s-fortress-1810.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/fyodor-bronnikov/cursed-field-the-place-of-execution-in-ancient-rome-crucified-slave-1878.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/fyodor-vasilyev/morning-1873-1.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/fyodor-vasilyev/pine-grove-in-the-swamp-1873.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/george-harvey/landscape-1874.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/george-lambert/bombay-1731.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/george-lambert/chiswick-house-middlesex-1742.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/george-lambert/landscape-with-figures-1757.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/gil-teixeira-lopes/espa-o-dourado.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/gustave-loiseau/banks-of-the-seine-in-summer-tournedos-sur-seine-1899.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/guy-rose/plums-waterglass-and-peaches-1889.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/henri-rousseau/seine-and-eiffel-tower-in-the-sunset-1910.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/isaac-levitan/quiet-cloister-1890-3.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ivan-aivazovsky/exploding-ship-1900.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ivan-aivazovsky/shipwreck-1854.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ivan-shishkin/forest-glade-glade-1897.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/jacob-isaakszoon-van-ruisdael/stormy-sea-with-sailing-vessels-1668.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/janos-tornyai/winter-landscape-with-violet-lights.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/john-glover/landscape-view-in-cumberland-1820.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/john-varley/a-view-of-the-thames-looking-towards-battersea.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/jules-breton/the-shepherd-1905.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/jules-dupre/forest-landscape-1840.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/jules-dupre/landscape-with-cattle-at-limousin-1837.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/konstantin-bogaevsky/evening-at-the-sea-1941.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/lembesis-polychronis/supreme-court-1880.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/lilla-cabot-perry/little-girl-n-a-lane-giverny-1907.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/marcus-larson/stemship-in-sunset.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/marianne-north/life-on-the-coast-of-praslin-seychelles-1883.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/maxim-vorobiev/italian-landscape-1847.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/maxime-maufra/morning-in-the-oasis-of-alkantra-1913.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/norman-ackroyd/study-of-sunlight.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/nutzi-acontz/landscape-with-river-1954.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/p-ricl-s-pantazis/supreme-court-1880.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/paul-bril/landscape-with-nymphs-and-satyrs-1623.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/pavel-svinyin/the-tomb-of-george-washington.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/pierre-puvis-de-chavannes/young-girls-by-the-sea.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/pieter-wenning/transvaal-evening-nelspruit.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/rosa-bonheur/sheep-in-a-landscape.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/sydney-laurence/mount-mckinley.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/sydney-laurence/the-trapper-1914.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/theodor-philipsen/long-shadows-cattle-on-the-island-of-saltholm-1890.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/theodor-severin-kittelsen/gutt-paa-hvit-hest.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/vasily-polenov/has-been-in-desert-1909.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/vasily-polenov/the-river-oka-1918.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/veniamin-kremer/cloudy-1975.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/volodymyr-orlovsky/at-the-field-1890.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/william-hart/harvest-scene-valley-of-the-delaware-1868.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/william-shayer/the-cowherd.jpg!Large.jpg"
      )
    )
    //    val styles = resolve(log)

    for ((styleKey, styleFiles) <- styles) {
      log.h1(styleKey)
      log.subreport(styleKey + "_images", (sub: NotebookOutput) => {
        for (url <- styleFiles) sub.p(sub.jpg(VisionPipelineUtil.load(url, -1), url))
        null
      })
      for (styleLayers <- VGG19.values().map(List(_)).toList) {
        log.h2(styleLayers.map(_.name()).mkString(", "))
        paintUsingLayers(styleLayers, styleFiles)(log)
      }
    }
    null
  }

  def paintUsingLayers(styleLayers: Seq[VisionPipelineLayer], styleFiles: Seq[String])(implicit log: NotebookOutput) = {
    def crossProductStyleNetwork(modifiers: VisualModifier*) = {
      (canvas: Tensor, viewLayer: Layer, precision: Precision, styleImages: Seq[String]) => {
        new CartesianStyleNetwork(
          styleLayers = styleLayers,
          styleModifiers = modifiers,
          styleUrl = styleImages,
          precision = precision,
          viewLayer = viewLayer,
          tileSize = TextureOperatorSurvey.this.tileSize,
          tilePadding = TextureOperatorSurvey.this.tilePadding,
          minWidth = TextureOperatorSurvey.this.styleMin,
          maxWidth = TextureOperatorSurvey.this.styleMax,
          maxPixels = TextureOperatorSurvey.this.stylePixelMax,
          magnification = TextureOperatorSurvey.this.styleMagnification
        ).apply(canvas)
      }
    }

    //    paint(crossProductStyleNetwork(log.eval(() => {
    //      List(
    //        new ChannelMeanMatcher()
    //      )
    //    }): _*), styleFiles)
    //    paint(crossProductStyleNetwork(log.eval(() => {
    //      List(
    //        new GramMatrixMatcher().setTileSize(tileSize)
    //      )
    //    }): _*), styleFiles)
    //    paint(crossProductStyleNetwork(log.eval(() => {
    //      List(
    //        new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(tileSize)
    //      )
    //    }): _*), styleFiles)
    paint(crossProductStyleNetwork(log.eval(() => {
      List(
        new ChannelMeanMatcher(),
        new GramMatrixMatcher().setTileSize(tileSize),
        new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(tileSize)
      )
    }): _*), styleFiles)
  }

  def paint(trainable: (Tensor, Layer, Precision, Seq[String]) => Trainable, styleFiles: Seq[String])(implicit log: NotebookOutput): Unit = {
    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas, trainable, styleFiles)(sub)
        null
      })
    }
  }

  def paint(canvas: AtomicReference[Tensor], trainable: (Tensor, Layer, Precision, Seq[String]) => Trainable, styleFiles: Seq[String])(implicit sub: NotebookOutput): Unit = {
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
      optimize(canvas.get, trainable(canvas.get, viewLayer, precision, styleFiles))
    }
  }

  def resolutions = (if (growth > 1) Stream.iterate(minResolution)(_ * growth).takeWhile(_ <= maxResolution) else List(minResolution, maxResolution).distinct).map(_.toInt).toArray

  private def growth = Math.pow(maxResolution / minResolution, 1.0 / resolutionSteps)

  def precision(w: Int) = if (w <= minResolution) Precision.Double else Precision.Float

  def resolve(log: NotebookOutput): Map[String, Array[String]] = {
    log.eval(() => {
      val map = List(
        "landscsapes" ->
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
            |cursed-field-the-place-of-execution-in-ancient-rome-crucified-slave-1878.jpg!Large.jpg""".stripMargin.split('\n').map(_.trim).toSet,
        "drawings" ->
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
            |king-and-his-subjects-2005.jpg!Large.jpg""".stripMargin.split('\n').map(_.trim).toSet,
        "abstract" ->
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
            |duel-1912.jpg!Large.jpg""".stripMargin.split('\n').map(_.trim).toSet
      ).map(t => {
        val (styleKey, files) = t
        styleKey -> findFiles(files)
      }).toMap
      println(ScalaJson.toJson(map))
      map
    })
  }

}

