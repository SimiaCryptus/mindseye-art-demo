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

package com.simiacryptus.mindseye.art.util

import java.awt.Desktop
import java.io.File
import java.net.URI
import java.nio.charset.Charset
import java.text.SimpleDateFormat
import java.util.{Date, UUID}

import org.apache.commons.io.{FileUtils, IOUtils}
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClientBuilder
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object NanoCrawl {
  def main(args: Array[String]): Unit = {
    val outputDir = new File(new SimpleDateFormat("yyyy_MM_dd_HH_mm").format(new Date()))
    outputDir.mkdirs()
    new NanoCrawl(outputDir, new URI(args.headOption.getOrElse("https://www.wikiart.org/en/"))).crawl()
    Desktop.getDesktop.open(outputDir)
  }
}

class NanoCrawl(file: File, base: URI) {
  lazy val logger = LoggerFactory.getLogger(classOf[NanoCrawl])
  lazy val client = HttpClientBuilder.create().build()

  val allowedExtensions = List(
    "htm", "html", "",
    "jpg", "jpeg", "gif", "png",
    "txt")
  val allowedHosts = List(base.getHost.split("\\.").takeRight(2).mkString("."))
  val visited = new ArrayBuffer[URI]()

  def crawl(url: URI = base): Unit = {
    if (!visited.contains(url)) try {
      logger.info(s"Fetching $url to $file")
      visited += url
      val response = client.execute(new HttpGet(url))
      lazy val responseEntity = response.getEntity
      val textSrc = IOUtils.toByteArray(responseEntity.getContent)
      response.close()

      val virtualPath = Option(url.getPath.stripPrefix("/").trim).filterNot(_.isEmpty).getOrElse(UUID.randomUUID().toString + ".html")
      val saveAs = new File(file, virtualPath + (if (virtualPath.contains(".")) "" else ".html"))
      logger.info(s"Saving $url to $saveAs")
      saveAs.getParentFile.mkdirs()
      FileUtils.writeByteArrayToFile(saveAs, textSrc)

      def charset = Charset.forName("UTF-8")

      responseEntity.getContentType.getValue.split(";").head match {
        case "text/html" =>
          val links = "(?i)(href|src)=\"([^\"]*)\"".r.findAllMatchIn(new String(textSrc, charset)).toList.map(x => Try {
            url.resolve(x.group(2).split("\\?").head.split("#").head)
          }).filter(_.isSuccess).map(_.get)
          val childLinks = (for (link <- links) yield {
            if (visited.contains(link)) {
              logger.debug(s"Redundant Link from $url: $link")
              None
            } else if (filter(link)) {
              logger.info(s"New Link from $url: $link")
              Option(link)
            } else {
              logger.info(s"Reject link from $url: $link")
              None
            }
          }).flatten.distinct
          logger.debug(s"All links from $url: $childLinks")
          val byExtension = childLinks.groupBy(x => Option(x.getPath).map(_.split("\\.").last).getOrElse("html"))
          (byExtension - "html").values.flatten.foreach(crawl(_))
          byExtension.get("html").toList.flatten.foreach(crawl(_))
        case other => logger.info(s"Cannot handle content type $other")
      }
    } catch {
      case e: Throwable => logger.warn(s"Error fetching $url", e)
    }
  }

  def filter(link: URI) = {
    val hostAllowed = link.getHost == null || allowedHosts.filter(s => link.getHost.endsWith(s)).headOption.isDefined
    val extension = Option(link.getPath).map(_.split("\\.").drop(1).lastOption.getOrElse("")).getOrElse("")
    hostAllowed && allowedExtensions.contains(extension.toLowerCase) && (link.getHost != base.getHost || (null != link.getPath && link.getPath.startsWith(base.getPath)))
  }
}
