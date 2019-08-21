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

import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.{CannedAccessControlList, ObjectMetadata, PutObjectRequest}
import com.simiacryptus.aws.EC2Util
import com.simiacryptus.sparkbook.NotebookRunner

class GifRegistration
(
  bucket: String,
  reportUrl: String,
  liveUrl: String,
  canvas: () => Seq[BufferedImage],
  instances: List[String] = List(
    EC2Util.instanceId()
  ).filterNot(_.isEmpty),
  id: String = UUID.randomUUID().toString,
  indexFile: String = "index.html",
  className: String = "",
  description: String = "",
  delay: Int = 100
) extends JobRegistration[Seq[BufferedImage]](bucket, reportUrl, liveUrl, canvas, instances, id, indexFile, className, description) {

  def uploadImage(canvas: Seq[BufferedImage])(implicit s3client: AmazonS3) = {
    val key = s"img/$id.gif"
    logger.info("Writing " + key)
    val metadata = new ObjectMetadata()
    val stream = new ByteArrayOutputStream()
    NotebookRunner.toGif(stream, canvas, delay)
    metadata.setContentType("image/gif")
    s3client.putObject(new PutObjectRequest(bucket, key, new ByteArrayInputStream(stream.toByteArray), metadata)
      .withCannedAcl(CannedAccessControlList.PublicRead))
    s"http://$bucket/$key"
  }

}

