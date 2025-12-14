import { useState, useEffect } from 'react'
import { API_URL } from '../config'

interface Analysis {
  id: string
  created_at: string
  people_count: number
  compliance_score: number
  violations: string[]
  report_generated?: boolean
  report_url?: string | null
  site_name?: string
  contractor_name?: string
  submitted_to?: string
}

export default function ReportsList() {
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generatingReport, setGeneratingReport] = useState<string | null>(null)
  
  // Form state for report metadata
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null)
  const [siteName, setSiteName] = useState('')
  const [contractorName, setContractorName] = useState('')
  const [submittedTo, setSubmittedTo] = useState('')
  const [showForm, setShowForm] = useState(false)

  useEffect(() => {
    fetchAnalyses()
  }, [])

  const fetchAnalyses = async () => {
    try {
      const response = await fetch(`${API_URL}/api/analyses`)
      if (!response.ok) throw new Error('Failed to fetch analyses')
      const data = await response.json()
      setAnalyses(data.analyses || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analyses')
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateReport = async (analysisId: string) => {
    setGeneratingReport(analysisId)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/api/analyses/${analysisId}/report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          site_name: siteName || 'Not specified',
          contractor_name: contractorName || 'Not specified',
          submitted_to: submittedTo || 'Not specified',
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to generate report')
      }

      const data = await response.json()
      
      // Download the report
      if (data.report_url) {
        const reportResponse = await fetch(`${API_URL}${data.report_url}`)
        const blob = await reportResponse.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `compliance-report-${analysisId.slice(0, 8)}.pdf`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
      }

      // Refresh analyses list
      await fetchAnalyses()
      
      // Reset form
      setShowForm(false)
      setSelectedAnalysis(null)
      setSiteName('')
      setContractorName('')
      setSubmittedTo('')
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate report')
    } finally {
      setGeneratingReport(null)
    }
  }

  const handleDownloadExisting = async (reportUrl: string, analysisId: string) => {
    try {
      const response = await fetch(`${API_URL}${reportUrl}`)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `compliance-report-${analysisId.slice(0, 8)}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError('Failed to download report')
    }
  }

  const openGenerateForm = (analysisId: string, analysis: Analysis) => {
    setSelectedAnalysis(analysisId)
    setSiteName(analysis.site_name || '')
    setContractorName(analysis.contractor_name || '')
    setSubmittedTo(analysis.submitted_to || '')
    setShowForm(true)
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#2ecc71'
    if (score >= 60) return '#f39c12'
    return '#e74c3c'
  }

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading analyses...</p>
      </div>
    )
  }

  if (error && analyses.length === 0) {
    return (
      <div className="error">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={fetchAnalyses} className="btn-secondary">
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="reports-container">
      <div className="reports-header">
        <h2>Compliance Reports</h2>
        <p className="reports-subtitle">
          Generate Build UK Standard Format PDF reports from your past analyses
        </p>
      </div>

      {error && (
        <div className="error-toast">
          {error}
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {showForm && selectedAnalysis && (
        <div className="modal-overlay" onClick={() => setShowForm(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Report Details</h3>
            <p className="modal-subtitle">Provide details for the compliance report</p>
            
            <div className="form-group">
              <label>Site Name</label>
              <input
                type="text"
                value={siteName}
                onChange={(e) => setSiteName(e.target.value)}
                placeholder="e.g., Canary Wharf Tower Site"
              />
            </div>

            <div className="form-group">
              <label>Contractor Name</label>
              <input
                type="text"
                value={contractorName}
                onChange={(e) => setContractorName(e.target.value)}
                placeholder="e.g., ABC Scaffolding Ltd"
              />
            </div>

            <div className="form-group">
              <label>Submitted To (Client)</label>
              <input
                type="text"
                value={submittedTo}
                onChange={(e) => setSubmittedTo(e.target.value)}
                placeholder="e.g., Multiplex Construction"
              />
            </div>

            <div className="modal-actions">
              <button
                onClick={() => setShowForm(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => handleGenerateReport(selectedAnalysis)}
                className="btn-primary"
                disabled={generatingReport === selectedAnalysis}
              >
                {generatingReport === selectedAnalysis ? 'Generating...' : 'Generate Report'}
              </button>
            </div>
          </div>
        </div>
      )}

      {analyses.length === 0 ? (
        <div className="empty-state">
          <h3>No analyses yet</h3>
          <p>Upload a construction site photo in the Photo Analysis tab to get started.</p>
        </div>
      ) : (
        <div className="reports-grid">
          {analyses.map((analysis) => (
            <div key={analysis.id} className="report-card">
              <div className="report-card-header">
                <div className="report-date">
                  {new Date(analysis.created_at).toLocaleString('en-GB', {
                    day: '2-digit',
                    month: 'short',
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
                <div
                  className="report-score"
                  style={{ color: getScoreColor(analysis.compliance_score) }}
                >
                  {analysis.compliance_score}/100
                </div>
              </div>

              <div className="report-card-body">
                <div className="report-stat">
                  <span className="stat-value">{analysis.people_count}</span>
                  <span className="stat-label">Workers</span>
                </div>
                <div className="report-stat">
                  <span className="stat-value">{analysis.violations.length}</span>
                  <span className="stat-label">Violations</span>
                </div>
              </div>

              <div className="report-card-footer">
                {analysis.report_generated && analysis.report_url ? (
                  <button
                    onClick={() => handleDownloadExisting(analysis.report_url!, analysis.id)}
                    className="btn-download"
                  >
                    ↓ Download Report
                  </button>
                ) : (
                  <button
                    onClick={() => openGenerateForm(analysis.id, analysis)}
                    className="btn-generate"
                    disabled={generatingReport === analysis.id}
                  >
                    {generatingReport === analysis.id ? 'Generating...' : 'Generate Report'}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

