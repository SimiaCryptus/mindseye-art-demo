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

trait GeometricResolutionSequence {

  def minResolution: Int

  def maxResolution: Int

  def resolutionSteps: Int

  def resolutions = Stream.iterate(minResolution.toDouble)(_ * growth).takeWhile(_ <= maxResolution).map(_.toInt)

  private def growth = Math.pow(maxResolution / minResolution, 1.0 / resolutionSteps)

}
