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

import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.test.TestUtil

import scala.util.Random

class ImageSource(urls: Seq[String]) {
  def tileSize: Int = 400

  def tilePadding: Int = 16

  def loadImages(canvasPixels: Int): Array[Tensor] = {
    val styles = Random.shuffle(urls.toList).map(styleUrl => {
      var styleImage = VisionPipelineUtil.load(styleUrl, -1)
      val stylePixels = styleImage.getWidth * styleImage.getHeight
      var finalWidth = if (canvasPixels > 0) (styleImage.getWidth * Math.sqrt((canvasPixels.toDouble / stylePixels) * magnification)).toInt else -1
      if (finalWidth < minWidth && finalWidth > 0) finalWidth = minWidth
      if (finalWidth > Math.min(maxWidth, styleImage.getWidth)) finalWidth = Math.min(maxWidth, styleImage.getWidth)
      val resized = TestUtil.resize(styleImage, finalWidth, true)
      Tensor.fromRGB(resized)
    }).toBuffer
    while (styles.map(_.getDimensions).map(d => d(0) * d(1)).sum > maxPixels) styles.remove(0)
    styles.toArray
  }

  def minWidth: Int = 1

  def maxWidth: Int = 2048

  def maxPixels: Double = 5e6

  def magnification: Double = 1.0

}
