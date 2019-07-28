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

import java.util.UUID

import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.{CannedAccessControlList, ObjectMetadata, PutObjectRequest}
import com.simiacryptus.aws.EC2Util
import com.simiacryptus.mindseye.lang.Tensor

class JpgRegistration
(
  bucket: String,
  reportUrl: String,
  liveUrl: String,
  canvas: () => Tensor,
  instances: List[String] = List(
    EC2Util.instanceId()
  ).filterNot(_.isEmpty),
  id: String = UUID.randomUUID().toString
) extends JobRegistration[Tensor](bucket, reportUrl, liveUrl, canvas, instances, id) {

  def uploadImage(canvas: Tensor)(implicit s3client: AmazonS3) = {
    val key = s"img/$id.jpg"
    logger.info("Writing " + key)
    val metadata = new ObjectMetadata()
    metadata.setContentType("image/jpeg")
    s3client.putObject(new PutObjectRequest(bucket, key, toStream(canvas.toImage), metadata)
      .withCannedAcl(CannedAccessControlList.PublicRead))
    s"http://$bucket/$key"
  }

}

