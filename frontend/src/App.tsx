import { useState, useEffect } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import ResultsDisplay from './components/ResultsDisplay'
import Sidebar from './components/Sidebar'
import { API_URL } from './config'
import './styles/App.css'

interface AnalysisResult {
  people_count: number
  signage_text: string
  violations: string[]
  detections: Array<{
    type: string
    confidence: number
    bbox: number[]
  }>
  compliance_score: number
  original_image: string
  annotated_image: string
    document_provided?: boolean
    document_id?: string
    document_name?: string
    document_requirements?: {
    ppe_requirements?: string[]
    access_requirements?: string[]
    equipment_requirements?: string[]
    personnel_limits?: any
    inspection_requirements?: string[]
    zone_requirements?: string[]
    other_requirements?: string[]
  }
  cross_check?: {
    compliant: string[]
    violations: Array<{
      requirement: string
      photo_evidence: string
      regulation: string
      severity: string
    }>
    warnings: Array<{
      issue: string
      reason: string
    }>
    regulatory_checks: {
      cdm_2015_compliant: boolean
      hse_compliant: boolean
      issues_found: string[]
    }
  }
}

function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleImageUpload = async (imageFile: File, documentId?: string) => {
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('image', imageFile)
    if (documentId) {
      formData.append('document_id', documentId)
    }

    try {
      // Use enhanced endpoint
      const response = await fetch(`${API_URL}/api/analyze-with-document`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Analysis failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
  }

  const handleSelectAnalysis = (analysis: any) => {
    // Convert Supabase analysis format to ResultsDisplay format
    setResult({
      people_count: analysis.people_count,
      signage_text: analysis.signage_text || '',
      violations: analysis.violations || [],
      detections: analysis.detections || [],
      compliance_score: analysis.compliance_score,
      original_image: analysis.original_image_url?.replace(/^https?:\/\/[^/]+/, '') || '',
      annotated_image: analysis.annotated_image_url?.replace(/^https?:\/\/[^/]+/, '') || '',
      document_provided: analysis.document_provided || false,
      document_id: analysis.document_id,
      document_name: analysis.document_name,
      document_requirements: analysis.document_requirements,
      cross_check: analysis.cross_check
    })
  }

  // Refresh sidebar when new analysis is completed
  useEffect(() => {
    if (result) {
      // Trigger sidebar refresh by dispatching a custom event
      window.dispatchEvent(new Event('analysis-completed'))
    }
  }, [result])

  return (
    <div className="app">
      <Header />
      
      <div className="app-layout">
        <Sidebar onSelectAnalysis={handleSelectAnalysis} />
        
        <main className="main-content">
          {!result && !loading && (
            <ImageUpload onUpload={handleImageUpload} />
          )}

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Analyzing image for safety compliance...</p>
            </div>
          )}

          {error && (
            <div className="error">
              <h2>Error</h2>
              <p>{error}</p>
              <button onClick={handleReset} className="btn-secondary">
                Try Again
              </button>
            </div>
          )}

          {result && (
            <ResultsDisplay result={result} onReset={handleReset} />
          )}
        </main>
      </div>

      <footer>
        <p>Built with YOLOv8 + Tesseract OCR • Interface.ai Technical Demo</p>
      </footer>
    </div>
  )
}

export default App

