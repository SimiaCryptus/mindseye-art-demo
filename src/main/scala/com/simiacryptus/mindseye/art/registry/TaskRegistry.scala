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

import com.simiacryptus.aws.EC2Util
import com.simiacryptus.mindseye.art.util.ArtSetup.{ec2client, s3client}
import com.simiacryptus.mindseye.art.util.ArtUtil
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.util.ImageUtil
import com.simiacryptus.notebook.NotebookOutput

trait TaskRegistry {

  def s3bucket: String
  def indexFile: String = "index.html"
  def className = getClass.getSimpleName.stripSuffix("$")
  def description = ""

  def registerWithIndexGIF(canvas: => Seq[BufferedImage], delay :Int = 100)(implicit log: NotebookOutput) = {
    val archiveHome = log.getArchiveHome
    if (!s3bucket.isEmpty && null != archiveHome) Option(new GifRegistration(
      bucket = s3bucket.split("/").head,
      reportUrl = "http://" + archiveHome.getHost + "/" + archiveHome.getPath.stripSuffix("/").stripPrefix("/") + "/" + log.getRoot.getName + "/" + log.getName + ".html",
      liveUrl = s"http://${EC2Util.publicHostname()}:1080/",
      canvas = () => {
        val list = canvas.filter(_ != null)
        if(list.isEmpty) list else {
          val maxWidth = list.map(_.getWidth).max
          list.map(ImageUtil.resize(_, maxWidth, true))
        }
      },
      indexFile = indexFile,
      delay = delay,
      className = className,
      description = description
    ).start()(s3client, ec2client)) else None
  }

  def registerWithIndexGIF2(canvas: => Seq[Tensor], delay :Int = 100)(implicit log: NotebookOutput) = {
    val archiveHome = log.getArchiveHome
    if (!s3bucket.isEmpty && null != archiveHome) Option(new GifRegistration(
      bucket = s3bucket.split("/").head,
      reportUrl = "http://" + archiveHome.getHost + "/" + archiveHome.getPath.stripSuffix("/").stripPrefix("/") + "/" + log.getRoot.getName + "/" + log.getName + ".html",
      liveUrl = s"http://${EC2Util.publicHostname()}:1080/",
      canvas = () => {
        val list = canvas.filter(_ != null).map(_.toImage)
        val maxWidth = list.map(_.getWidth).max
        ArtUtil.cyclical(list.map(ImageUtil.resize(_, maxWidth, true)))
      },
      indexFile = indexFile,
      delay = delay,
      className = className,
      description = description
    ).start()(s3client, ec2client)) else None
  }

  def registerWithIndexJPG(canvas: => Tensor)(implicit log: NotebookOutput): Option[JobRegistration[Tensor]] = {
    val archiveHome = log.getArchiveHome
    if (!s3bucket.isEmpty && null != archiveHome) Option(new JpgRegistration(
      bucket = s3bucket.split("/").head,
      reportUrl = "http://" + archiveHome.getHost + "/" + archiveHome.getPath.stripSuffix("/").stripPrefix("/") + "/" + log.getRoot.getName + "/" + log.getName + ".html",
      liveUrl = s"http://${EC2Util.publicHostname()}:1080/",
      canvas = () => canvas,
      indexFile = indexFile,
      className = className,
      description = description
    ).start()(s3client, ec2client)) else None
  }

}
