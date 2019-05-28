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
import com.simiacryptus.mindseye.art.models.PoolingPipeline._
import com.simiacryptus.mindseye.art.models.VGG19
import com.simiacryptus.mindseye.art.ops.{ChannelMeanMatcher, ContentMatcher, GramMatrixEnhancer, GramMatrixMatcher}
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.{CudaSettings, Precision}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{NotebookOutput, NullNotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner.withMonitoredJpg
import com.simiacryptus.sparkbook._
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object ContentZoomDemo_EC2 extends ContentZoomDemo with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]"
  )

}

object ContentZoomDemo_Local extends ContentZoomDemo with LocalRunner[Object] with NotebookRunner[Object] {
  override def inputTimeoutSeconds = 5
}

class ContentZoomDemo extends ArtSetup[Object] with BasicOptimizer {
  val contentUrl = "file:///C:/Users/andre/Downloads/IMG_20190422_150855449.jpg"

  override def maxRate = 1e8

  override def trainingIterations = 50

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    log.eval(() => {
      ScalaJson.toJson(Map(
        "this" -> ContentZoomDemo.this
      ))
    })
    val canvas = new AtomicReference[Tensor](null)
    withMonitoredJpg(() => Option(canvas.get()).map(_.toRgbImage).orNull) {
      log.subreport(UUID.randomUUID().toString, (sub: NotebookOutput) => {
        paint(canvas)(sub)
        null
      })
      null
    }(log)
    null
  }

  def paint(canvas: AtomicReference[Tensor])(implicit sub: NotebookOutput): Unit = {
    val res = 600
    val precision = Precision.Double
    CudaSettings.INSTANCE().defaultPrecision = precision
    val tileSize = 300
    val network = new CartesianStyleNetwork(
      styleLayers = List(
        VGG19.VGG19_1b2,
        VGG19.VGG19_1c1,
        VGG19.VGG19_1c2,
        VGG19.VGG19_1c3,
        VGG19.VGG19_1c4
      ),
      styleModifiers = List(
        new ChannelMeanMatcher(),
        new GramMatrixMatcher().setTileSize(tileSize),
        new GramMatrixEnhancer().setMinMax(-0.25, 0.25).setTileSize(tileSize)
      ),
      styleUrl = List(
        "s3a://data-cb03c/crawl/wikiart/images/abraham-manievich/birch-trees-1911.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/aladar-korosfoi-kriesch/pupppet-theatre-1907.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/alice-neel/hartley-ginny-1970.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/alphonse-mucha/job-1896.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/anna-ostroumova-lebedeva/spring-motif-view-from-stone-island-to-krestovsky-and-yelagin-islands-1904.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/audrey-flack/royal-flush-1977.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/aurel-cojan/mouvement-1989.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ende/preliminaries-the-alpha-the-maiestas-domini-and-the-portraits-of-the-authors.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ende/the-woman-garbed-by-the-sun-and-the-dragon.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/francois-arnal/untitled-1949.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/henri-rousseau/negro-attacked-by-a-jaguar-1910.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/john-bratby/girl-with-a-rose-in-her-lap-1960.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/karl-bryullov/walk-of-louis-xv-in-childhood.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/kitagawa-utamaro/illustration-from-the-twelve-hours-of-the-green-houses-c-1795-colour-woodblock-print.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/kitagawa-utamaro/not_detected_233129.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/kitagawa-utamaro/the-hour-of-the-dragon.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/m-c-escher/metamorphosis-iii-1968-2.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/otto-eckmann/untitled-1899.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/twins-seven-seven/king-and-his-subjects-2005.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/unichi-hiratsuka/plum-1930.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/ustad-mansur/a-barbet-himalayan-blue-throated-bird-1615.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/utagawa-kunisada-ii/the-dragon.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/utagawa-toyokuni-ii/the-kabuki-actors-ichikawa-danjuro-vii-as-iwafuji-1824.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/william-morris/design-for-tulip-and-willow-indigo-discharge-wood-block-printed-fabric-1873.jpg!Large.jpg", "s3a://data-cb03c/crawl/wikiart/images/wladyslaw-strzeminski/d-landscape-1932.jpg!Large.jpg"
      ),
      precision = precision,
      maxPixels = 1e7,
      tileSize = tileSize
    ).withContent(List(
      Pooling16
    ), List(
      new ContentMatcher().pow(2).scale(1e5)
    ))
    val downRes = 16
    val content = VisionPipelineUtil.load(contentUrl, res)
    if (null == canvas.get) {
      implicit val nullNotebookOutput = new NullNotebookOutput()
      canvas.set(Tensor.fromRGB(TestUtil.resize(
        VisionPipelineUtil.load(contentUrl, content.getWidth / downRes, content.getHeight / downRes),
        content.getWidth, content.getHeight
      )))
    }
    else {
      canvas.set(Tensor.fromRGB(TestUtil.resize(canvas.get.toRgbImage, content.getWidth, content.getHeight)))
    }
    optimize(canvas.get, network.copy(precision = precision).apply(canvas.get, Tensor.fromRGB(content)))
  }

}


