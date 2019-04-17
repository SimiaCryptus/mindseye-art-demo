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

case class Permutation(indices: Array[Int]) {

  val unity = Permutation((1 to indices.length).toArray)

  def ^(n: Int) = {
    Stream.iterate(unity)(this * _)(n)
  }

  def *(right: Permutation) = {
    Permutation(this * right.indices)
  }

  def *(right: Array[Int]) = {
    indices.map(idx => {
      if (idx < 0) {
        -right(-idx - 1)
      } else {
        right(idx - 1)
      }
    })
  }

  override def toString: String = "[" + indices.mkString(",") + "]"

  override def hashCode(): Int = indices.toList.hashCode()

  override def equals(obj: scala.Any): Boolean = obj match {
    case obj: Permutation => indices.toList.equals(obj.indices.toList)
    case _ => false
  }
}
