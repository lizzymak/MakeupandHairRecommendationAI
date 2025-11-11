import React, { useState, type ChangeEvent, type FormEvent } from 'react'
import './App.css'

interface Result{
  face_shape: string
  eye_shape: string
  undertone: string
  recommended_palette: string[]
}

function App() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<Result | null>(null)

  const fileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    if(e.target.files){
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if(!file) return

    const formData = new FormData
    formData.append("file", file)
    const res = await fetch("http://localhost:5000/analyze2", {
      method: "POST",
      body: formData
    })
    const data: Result = await res.json()
    console.log(result);
    console.log(data);
    setResult(data)
  }

  return (
    <div>
      <h2>Upload Image</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={fileUpload} />
        <button type="submit">Analyze</button>
      </form>
      {result && (
        <div>
          <p>Face shape: {result.face_shape}</p>
          <p>Eye shape: {result.eye_shape}</p>
          <p>Skin undertone: {result.undertone}</p>
          <div>
          {result?.recommended_palette?.map(color => (
            <div key={color} style={{ width: 50, height: 50, backgroundColor: color }}></div>
          ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
