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

import com.amazonaws.services.ec2.AmazonEC2
import com.amazonaws.services.ec2.model.DescribeInstanceStatusRequest
import com.amazonaws.services.s3.AmazonS3
import com.amazonaws.services.s3.model.ListObjectsRequest
import com.simiacryptus.sparkbook.util.ScalaJson
import org.apache.commons.io.IOUtils

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.Try

object JobRegistry {

  def list(bucket: String)(implicit s3client: AmazonS3): Seq[JobRegistry] = {
    val objectListing = s3client.listObjects(new ListObjectsRequest().withBucketName(bucket).withPrefix("jobs/"))
    for (item <- objectListing.getObjectSummaries) yield {
      ScalaJson.fromJson(IOUtils.toString(s3client.getObject(bucket, item.getKey).getObjectContent, "UTF-8"), classOf[JobRegistry])
    }
  }

}

case class JobRegistry
(
  reportUrl: String,
  liveUrl: String,
  lastReport: Long,
  instances: List[String],
  image: String,
  id: String,
  className: String,
  description: String
) {

  def save(bucket: String)(implicit s3client: AmazonS3) = {
    val key = s"jobs/$id.json"
    s3client.putObject(bucket, key, ScalaJson.toJson(this).toString)
    s"s3://$bucket/$key"
  }

  def isLive()(implicit ec2client: AmazonEC2) = Try {
    !runningInstances(ec2client).isEmpty
  }

  def runningInstances(implicit ec2client: AmazonEC2) = {
    val instances = this.instances.filterNot(_.isEmpty)
    ec2client.describeInstanceStatus(new DescribeInstanceStatusRequest()
      .withInstanceIds(instances: _*))
      .getInstanceStatuses.asScala
      .filter(x => instances.contains(x.getInstanceId))
      .filter(_.getInstanceState.getName == "running")
      .toList
  }
}
