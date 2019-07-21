import React, { useState } from 'react'
import logo from './logo.svg'
import './App.css'
import * as tf from '@tensorflow/tfjs'

// known from model
const corpusSize = 251
const nextWords = 45

// Look of word to index
const word2index = require('./words.json')
const index2word = require('./lookup.json')

// In goes lowercase text, out goes a tensor that looks like this:
// [[0, 0, 0, ..., 7, 25, 102]]
const textToSequences = inText => {
  const words = inText.toLowerCase().split(' ')
  const wordIndexes = words.map(w => word2index[w])
  let indexList = tf.tensor1d(wordIndexes).toInt()
  // Pad the front with 0s
  const leftovers = corpusSize - wordIndexes.length
  return indexList.pad([[leftovers, 0]]).reshape([1, 251])
}

// this is where the model will go
let model

function App() {
  const [running, setRunning] = useState(false)
  const [poetrySeed, setPoetrySeed] = useState(null)
  const [poetry, setPoetry] = useState(null)

  const load = async () => {
    // this is a Layers Model
    model = await tf.loadLayersModel('/tfjs_quant_model/model.json')
  }

  load()

  // YES, I KNOW this could be optimized
  // GEEZ MOM!  STAHP!
  const createPoetry = async () => {
    if (!poetrySeed) return
    setRunning(true)
    let makePoetry = poetrySeed
    for (let i = 0; i < nextWords; i++) {
      const inputSeq = textToSequences(makePoetry)
      const predictions = await model.predict(inputSeq).data()
      inputSeq.dispose()
      const resultIdx = tf.argMax(predictions).dataSync()
      makePoetry += ' ' + index2word[resultIdx - 1]
      setPoetry(makePoetry)
    }
    setRunning(false)
  }

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h2 className="poetryTag">Poetry</h2>
        <p>{running ? 'Nom nom nom, eating GPUs and working' : ''}</p>
        <input
          key="seed"
          className="form inputLine"
          placeholder="Start your poem here"
          type="text"
          onBlur={({ target }) => setPoetrySeed(target.value)}
        />
        <button
          className={running ? 'form shiny' : 'form' }
          type="submit"
          disabled={running}
          onClick={createPoetry}
        >
          Generate Beat/Slam Poetry
        </button>
        <p className="results">{poetry}</p>
      </header>
    </div>
  )
}

export default App
