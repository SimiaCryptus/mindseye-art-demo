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

import org.apache.commons.math3.linear._

import scala.util.Random

object Permutation {
  def main(args: Array[String]): Unit = {
    rings(3).values.flatten.toList.sortBy(_.size).foreach(ring => {
      println(ring.map(_.toString).mkString(" -> "))
    })
  }

  def rings(rank: Int) = permutations(rank).map(_.ring).groupBy(_.size).mapValues(
    _.sortBy(_.map(_.indices.mkString(",")).mkString(";"))
      .groupBy(_.map(_.indices.mkString(",")).sorted.mkString(";"))
      .values.map(_.head).toSet)

  def permutations(rank: Int) = {
    (1 to rank).toStream.map(i => Stream(List(-i), List(i))).reduce((xs, ys) => {
      for {x <- xs; y <- ys} yield (x ++ y)
    }).flatMap(_.permutations).map(Permutation(_: _*))
  }

  def apply(indices: Int*) = new Permutation(indices.toArray)

  def roots(rank: Int, power: Int) = Random.shuffle(rings(rank)(power).flatMap(_.dropRight(1)).toStream)

  def unity(n: Int) = Permutation((1 to n).toArray: _*)
}

class Permutation(val indices: Array[Int]) {
  require(indices.distinct.size == indices.size)
  require(indices.map(_.abs).distinct.size == indices.size)
  require(indices.map(_.abs).min == 1)
  require(indices.map(_.abs).max == indices.size)

  def ^(n: Int): Permutation = Stream.iterate(unity)(this * _)(n)

  def unity = Permutation.unity(rank)

  def rank: Int = indices.length

  def *(right: Permutation): Permutation = Permutation(this * right.indices: _*)

  def *(right: Array[Int]) = {
    indices.map(idx => {
      if (idx < 0) {
        -right(-idx - 1)
      } else {
        right(idx - 1)
      }
    })
  }

  def matrix: RealMatrix = {
    val rank = this.rank
    val matrix = new Array2DRowRealMatrix(3, 3)
    val tuples = indices.zipWithIndex.map(t => (t._1.abs - 1, t._2, t._1.signum))
    for ((x, y, v) <- tuples) matrix.setEntry(x, y, v)
    matrix
  }

  def ring = {
    List(this) ++ Stream.iterate(this)(_ * this).drop(1).takeWhile(_ != this)
  }

  override def toString: String = "[" + indices.mkString(",") + "]"

  override def equals(obj: scala.Any): Boolean = obj match {
    case obj: Permutation if (obj.hashCode() == this.hashCode()) => indices.toList.equals(obj.indices.toList)
    case _ => false
  }

  override def hashCode(): Int = indices.toList.hashCode()
}
