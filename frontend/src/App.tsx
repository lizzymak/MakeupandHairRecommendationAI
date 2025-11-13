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
  const [imgPreview, setImagePreview] = useState<string | "">("")
  const [loading, setLoading] = useState<boolean | false>(false)

  const fileUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if(!file) return
    setFile(file)
    const reader = new FileReader()
    reader.onloadend = () => {
      if(typeof reader.result === "string"){
        setImagePreview(reader.result)
      }
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if(!file) return
    setLoading(true)
    const formData = new FormData
    formData.append("file", file)
    try{
      const res = await fetch("http://localhost:5000/analyze2", {
      method: "POST",
      body: formData
    })
    const data: Result = await res.json()
    setResult(data)
    }
    catch(err){
      console.log(err)
    }
    finally{
      setLoading(false)
    }
  }

  return (
    <div>
      <h2>Face Analyzer</h2>
      <div className='mainLayout'>
        <div className='uploadArea'>
          {imgPreview ? (
                <img src={imgPreview} alt="Upload image" style={{width: "300px"}}/>):(<p>Upload Image...</p>)}
          <form onSubmit={handleSubmit}>
            <input type="file" onChange={fileUpload} />
            <button type="submit">Analyze</button>
          </form>
        </div>
        <div>
          {loading && <p>Analyzing image...</p>}
          {result && !loading && (
          <div>
            <div className='categoryLine'>
              <p className='categoryBox'>Face Shape</p>
              <p>{result.face_shape}</p>
            </div>

            <div className='categoryLine'>
              <p className='categoryBox'>Eye Shape</p>
              <p>{result.eye_shape}</p>
            </div>

            <div className='categoryLine'>
              <p className='categoryBox'>Skin Undertone</p>
              <p>{result.undertone}</p>
            </div>
            
            <div className='gridContainer'>
            {result?.recommended_palette?.map(color => (
              <div key={color} style={{ width: 50, height: 50, backgroundColor: color }}></div>
            ))}
            </div>
          </div>)}
        </div>
        
    </div>
    </div>
  )
}

export default App
