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

package com.simiacryptus.mindseye.art.registry

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.util.UUID
import java.util.concurrent.{Executors, ScheduledExecutorService, ScheduledFuture, TimeUnit}

import com.amazonaws.services.ec2.AmazonEC2
import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.{CannedAccessControlList, ObjectMetadata, PutObjectRequest}
import com.simiacryptus.aws.EC2Util
import com.simiacryptus.sparkbook.util.Logging
import javax.imageio.ImageIO

object JobRegistration {
  lazy val scheduledExecutorService: ScheduledExecutorService = Executors.newScheduledThreadPool(1)
}

import com.simiacryptus.mindseye.art.registry.JobRegistration._

abstract class JobRegistration[T]
(
  bucket: String,
  reportUrl: String,
  liveUrl: String,
  canvas: () => T,
  instances: List[String] = List(
    EC2Util.instanceId()
  ).filterNot(_.isEmpty),
  id: String = UUID.randomUUID().toString
) extends AutoCloseable with Logging {
  var future: ScheduledFuture[_] = null

  def start()(implicit s3client: AmazonS3, ec2client: AmazonEC2) = {
    future = scheduledExecutorService.scheduleAtFixedRate(new Runnable {
      override def run(): Unit = try {
        update()
      } catch {
        case e: Throwable => logger.warn("Error updating " + JobRegistration.this, e)
      }
    }, periodMinutes, periodMinutes, TimeUnit.MINUTES)
    this
  }

  def periodMinutes = 5

  def update()(implicit s3client: AmazonS3, ec2client: AmazonEC2) = {
    upload()
    rebuildIndex()
  }

  def upload()(implicit s3client: AmazonS3) = {
    logger.info("Writing " + JobRegistry(
      reportUrl = reportUrl,
      liveUrl = liveUrl,
      lastReport = System.currentTimeMillis(),
      instances = List(EC2Util.instanceId()),
      image = uploadImage(canvas()),
      id = id
    ).save(bucket))
  }

  def rebuildIndex()(implicit s3client: AmazonS3, ec2client: AmazonEC2) = {
    val bodyHtml = (for (item <- JobRegistry.list(bucket)) yield {
      if (item.isLive()(ec2client).toOption.getOrElse(false)) {
        s"""<div><a href="${item.liveUrl}"><img src="${item.image}" style="max-width: 50%" /></a></div>""".stripMargin
      } else {
        s"""<div><a href="${item.reportUrl}"><img src="${item.image}" style="max-width: 50%" /></a></div>""".stripMargin
      }
    }).mkString("\n")
    logger.info(s"Writing http://$bucket/index.html")
    val metadata = new ObjectMetadata()
    metadata.setContentType("text/html")
    s3client.putObject(new PutObjectRequest(bucket, "index.html", new ByteArrayInputStream(
      s"""<html><head></head><body>
         |$bodyHtml
         |</body></html>""".stripMargin.getBytes
    ), metadata).withCannedAcl(CannedAccessControlList.PublicRead))
  }

  def stop()(implicit s3client: AmazonS3, ec2client: AmazonEC2) = {
    update()
    close()
  }

  override def close(): Unit = {
    if (null != future) {
      future.cancel(false)
      future = null
    }
  }

  def uploadImage(value: T)(implicit s3client: AmazonS3): String

  def toStream(image: BufferedImage): ByteArrayInputStream = {
    val outputStream = new ByteArrayOutputStream()
    ImageIO.write(image, "jpg", outputStream)
    new ByteArrayInputStream(outputStream.toByteArray)
  }
}
