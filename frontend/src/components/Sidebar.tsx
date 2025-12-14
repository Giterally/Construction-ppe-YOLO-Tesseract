import { useState, useEffect } from 'react'
import { API_URL } from '../config'

interface PastAnalysis {
  id: string
  people_count: number
  compliance_score: number
  violations: string[]
  created_at: string
  original_image_url: string
  annotated_image_url: string
  original_image?: string
  annotated_image?: string
  signage_text?: string
  detections?: Array<{
    type: string
    confidence: number
    bbox: number[]
  }>
  ocr_processing_steps?: Array<{
    step: number | string
    name: string
    status: string
    [key: string]: any
  }>
  [key: string]: any
}

interface SidebarProps {
  onSelectAnalysis: (analysis: PastAnalysis) => void
  selectedAnalysisId?: string
}

export default function Sidebar({ onSelectAnalysis, selectedAnalysisId }: SidebarProps) {
  const [analyses, setAnalyses] = useState<PastAnalysis[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchAnalyses()
    
    // Listen for new analysis completion
    const handleAnalysisComplete = () => {
      fetchAnalyses()
    }
    
    window.addEventListener('analysis-completed', handleAnalysisComplete)
    return () => {
      window.removeEventListener('analysis-completed', handleAnalysisComplete)
    }
  }, [])

  const fetchAnalyses = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/api/analyses?limit=20`)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Failed to fetch analyses (${response.status})`)
      }

      const data = await response.json()
      setAnalyses(data.analyses || [])
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load past analyses'
      setError(errorMessage)
      console.error('Error fetching analyses:', err)
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'var(--color-success)'
    if (score >= 60) return 'var(--color-warning)'
    return 'var(--color-error)'
  }

  const handleDelete = async (analysisId: string, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent triggering the onClick for selecting the analysis
    
    if (!confirm('Are you sure you want to delete this analysis?')) {
      return
    }

    try {
      const response = await fetch(`${API_URL}/api/analyses/${analysisId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || 'Failed to delete analysis')
      }

      // Refresh the list after deletion
      fetchAnalyses()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete analysis')
      console.error('Error deleting analysis:', err)
    }
  }

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to delete ALL past analyses? This action cannot be undone.')) {
      return
    }

    try {
      const response = await fetch(`${API_URL}/api/clear-all-analyses`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || 'Failed to delete all analyses')
      }

      // Refresh the list after deletion
      fetchAnalyses()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete all analyses')
      console.error('Error deleting all analyses:', err)
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Past Analyses</h2>
        <div className="sidebar-header-actions">
          <button onClick={fetchAnalyses} className="btn-refresh" title="Refresh">
            ↻
          </button>
          {analyses.length > 0 && (
            <button onClick={handleClearAll} className="btn-clear-all" title="Clear all analyses">
              Clear All
            </button>
          )}
        </div>
      </div>

      <div className="sidebar-content">
        {loading && (
          <div className="sidebar-loading">
            <div className="spinner-small"></div>
            <p>Loading...</p>
          </div>
        )}

        {error && (
          <div className="sidebar-error">
            <p>{error}</p>
            <button onClick={fetchAnalyses} className="btn-link">
              Retry
            </button>
          </div>
        )}

        {!loading && !error && analyses.length === 0 && (
          <div className="sidebar-empty">
            <p>No past analyses yet</p>
            <p className="sidebar-empty-hint">Upload an image to get started</p>
          </div>
        )}

        {!loading && !error && analyses.length > 0 && (
          <div className="analyses-list">
            {analyses.map((analysis) => (
              <div
                key={analysis.id}
                className={`analysis-item ${selectedAnalysisId === analysis.id ? 'selected' : ''}`}
                onClick={() => onSelectAnalysis(analysis)}
              >
                <div className="analysis-item-header">
                  <span className="analysis-date">{formatDate(analysis.created_at)}</span>
                  <div className="analysis-header-right">
                    <span
                      className="analysis-score"
                      style={{ color: getScoreColor(analysis.compliance_score) }}
                    >
                      {analysis.compliance_score}
                    </span>
                    <button
                      className="analysis-delete-btn"
                      onClick={(e) => handleDelete(analysis.id, e)}
                      title="Delete this analysis"
                      aria-label="Delete analysis"
                    >
                      ×
                    </button>
                  </div>
                </div>
                <div className="analysis-item-details">
                  <span className="analysis-people">
                    {analysis.people_count} {analysis.people_count === 1 ? 'person' : 'people'}
                  </span>
                  {analysis.violations.length > 0 && (
                    <span className="analysis-violations">
                      {analysis.violations.length} {analysis.violations.length === 1 ? 'violation' : 'violations'}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

