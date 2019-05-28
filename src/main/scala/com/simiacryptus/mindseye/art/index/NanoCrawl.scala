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

package com.simiacryptus.mindseye.art.index

import java.net.URI
import java.nio.charset.Charset
import java.util.UUID

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.art.util.VisionPipelineUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.util.{LocalRunner, Logging}
import com.simiacryptus.sparkbook.{EC2Runner, InteractiveSetup, NotebookRunner}
import org.apache.commons.io.IOUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClientBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Try

object NanoCrawl {
  lazy val client = HttpClientBuilder.create().build()
}

import com.simiacryptus.mindseye.art.index.NanoCrawl._


object NanoCrawl_EC2 extends EC2Runner[Object] with NanoCrawl {
  override val base: String = "https://www.wikiart.org/en/"
  override val storage: String = s"s3a://$s3bucket/crawl/wikiart/"

  override def sparkFactory: SparkSession = SparkSession.builder().master("local[4]").getOrCreate()

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.T2_L
}

object NanoCrawl_Local extends LocalRunner[Object] with NanoCrawl {
  override val base: String = "https://www.wikiart.org/en/"
  override val storage: String = "file:///E:/data/crawl/"

  override def http_port = 1081

  override def sparkFactory: SparkSession = {
    val builder = SparkSession.builder()
    import scala.collection.JavaConverters._
    VisionPipelineUtil.getHadoopConfig().asScala.foreach(t => builder.config(t.getKey, t.getValue))
    builder.master("local[2]").getOrCreate()
  }
}

trait NanoCrawl extends InteractiveSetup[Object] with NotebookRunner[Object] with Logging {
  private lazy val allowedHosts = List(new URI(base).getHost.split("\\.").takeRight(2).mkString(".")).toArray
  val allowedExtensions = Array(
    "htm", "html", "",
    "jpg", "jpeg", "gif", "png",
    "txt")

  def base: String

  def storage: String

  @JsonIgnore def sparkFactory: SparkSession

  override def postConfigure(log: NotebookOutput): Object = {
    implicit val spark = sparkFactory
    var uris: RDD[URI] = spark.sparkContext.parallelize(List(new URI(base)))
    var newAddrs = uris.flatMap(process(_)).distinct().subtract(uris).cache()
    while (!newAddrs.isEmpty()) {
      uris = newAddrs.union(uris)
      newAddrs = newAddrs.flatMap(process(_)).distinct().subtract(uris).cache()
      log.p("Fetched " + newAddrs.count())
    }
    log.p("Done")
    null
  }

  def process(url: URI): Seq[URI] = {
    try {
      val (textSrc: Array[Byte], mimeType) = fetch(url)
      save(url, textSrc)
      mimeType.flatMap({
        case "text/html" =>
          getLinks(url, textSrc)
        case other =>
          logger.debug(s"Cannot handle content type $other")
          List.empty
      })
    } catch {
      case e: Throwable =>
        logger.warn("Error in " + url, e)
        List.empty
    }
  }


  def getLinks(url: URI, html: Array[Byte]): List[URI] = {
    lazy val charset = Charset.forName("UTF-8")
    val links = "(?i)(href|src)=\"([^\"]*)\"".r.findAllMatchIn(new String(html, charset)).toList.map(x => Try {
      url.resolve(x.group(2).split("\\?").head.split("#").head)
    }).filter(_.isSuccess).map(_.get)
    val childLinks = (for (link <- links) yield {
      if (filter(link)) {
        logger.debug(s"New Link from $url: $link")
        Option(link)
      } else {
        logger.debug(s"Reject link from $url: $link")
        None
      }
    }).flatten.distinct
    logger.debug(s"All links from $url: $childLinks")
    childLinks
  }

  def filter(link: URI): Boolean = {
    val hostAllowed = link.getHost == null || allowedHosts.filter(s => link.getHost.endsWith(s)).headOption.isDefined
    val extension = Option(link.getPath).map(_.split("\\.").drop(1).lastOption.getOrElse("")).getOrElse("")
    hostAllowed && allowedExtensions.contains(extension.toLowerCase) && (link.getHost != new URI(base).getHost || (null != link.getPath && link.getPath.startsWith(new URI(base).getPath)))
  }

  def save(url: URI, data: Array[Byte]) = {
    if (url.getPath.split('.').last.toLowerCase match {
      case "html" => false
      case "htm" => false
      case "" => false
      case _ => true
    }) {
      val fileSystem = FileSystem.get(new URI(storage), VisionPipelineUtil.getHadoopConfig())
      val virtualPath = Option(url.getPath.stripPrefix("/").trim).filterNot(_.isEmpty).getOrElse(UUID.randomUUID().toString + ".html")
      val saveTarget = new Path(storage, virtualPath + (if (virtualPath.contains(".")) "" else ".html"))
      val saveAs = fileSystem.create(saveTarget)
      logger.info(s"Saving $url to $saveTarget")
      IOUtils.write(data, saveAs)
      saveAs.close()
    }
  }

  def fetch(url: URI) = {
    logger.debug(s"Fetching $url")
    val response = client.execute(new HttpGet(url))
    lazy val responseEntity = response.getEntity
    val textSrc = IOUtils.toByteArray(responseEntity.getContent)
    val mimeType = responseEntity.getContentType.getValue.split(";")
    response.close()
    (textSrc, mimeType)
  }

}
