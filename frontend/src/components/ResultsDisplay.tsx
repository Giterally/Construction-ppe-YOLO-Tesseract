import { API_URL } from '../config'

interface ResultsDisplayProps {
  result: {
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
    ocr_processing_steps?: Array<{
      step: number | string
      name: string
      status: string
      [key: string]: any
    }>
  }
  onReset: () => void
}

export default function ResultsDisplay({ result, onReset }: ResultsDisplayProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return '#2ecc71'
    if (score >= 60) return '#f39c12'
    return '#e74c3c'
  }

  return (
    <div className="results">
      <div className="results-header">
        <h2>Analysis Results</h2>
        <button onClick={onReset} className="btn-secondary">
          Analyze New Image
        </button>
      </div>

      {/* Compliance Score */}
      <div className="score-card" style={{ borderLeftColor: getScoreColor(result.compliance_score) }}>
        <div className="score-content">
          <h3>Compliance Score</h3>
          <div className="score-number" style={{ color: getScoreColor(result.compliance_score) }}>
            {result.compliance_score}/100
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{result.people_count}</div>
          <div className="metric-label">Workers Detected</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{result.detections.length}</div>
          <div className="metric-label">Total Detections</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{result.violations.length}</div>
          <div className="metric-label">Potential Violations</div>
        </div>
      </div>

      {/* Violations */}
      {result.violations.length > 0 && (
        <div className="violations-section">
          <h3>⚠️ Potential Violations</h3>
          <ul className="violations-list">
            {result.violations.map((violation, index) => (
              <li key={index}>{violation}</li>
            ))}
          </ul>
        </div>
      )}

      {result.violations.length === 0 && (
        <div className="no-violations">
          <h3>✓ No violations detected</h3>
          <p>Image analysis shows no immediate compliance concerns.</p>
        </div>
      )}

      {/* Signage Text */}
      {result.signage_text && (
        <div className="signage-section">
          <h3>Detected Signage Text</h3>
          <div className="signage-text">
            {result.signage_text || 'No text detected'}
          </div>
        </div>
      )}

      {/* Image Comparison */}
      <div className="image-comparison">
        <div className="image-container">
          <h3>Original Image</h3>
          <img
            src={`${API_URL}${result.original_image}`}
            alt="Original"
          />
        </div>
        <div className="image-container">
          <h3>Detected Objects</h3>
          <img
            src={`${API_URL}${result.annotated_image}`}
            alt="Annotated"
          />
        </div>
      </div>

      {/* Detection Details */}
      <details className="details-section">
        <summary>Detection Details ({result.detections.length} objects)</summary>
        <div className="detections-list">
          {result.detections.map((detection, index) => (
            <div key={index} className="detection-item">
              <span className="detection-type">{detection.type}</span>
              <span className="detection-confidence">
                {(detection.confidence * 100).toFixed(1)}% confidence
              </span>
            </div>
          ))}
        </div>
      </details>

      {/* OCR Processing Steps - Visual Timeline */}
      {result.ocr_processing_steps && result.ocr_processing_steps.length > 0 && (
        <details className="details-section ocr-steps">
          <summary>OCR Processing Steps</summary>
          <div className="ocr-visual-timeline">
            {result.ocr_processing_steps
              .sort((a, b) => {
                // Sort by step number if both are numbers, otherwise maintain order
                const aStep = typeof a.step === 'number' ? a.step : 999
                const bStep = typeof b.step === 'number' ? b.step : 999
                return aStep - bStep
              })
              .map((step, index) => (
              <div key={index} className="ocr-step-visual">
                <div className="ocr-step-title">{step.name}</div>
                {step.image && (
                  <div className="ocr-step-image-container">
                    <img 
                      src={`${API_URL}${step.image}`} 
                      alt={step.name}
                      className="ocr-step-image"
                    />
                  </div>
                )}
                {step.highlighted_image && (
                  <div className="ocr-step-image-container">
                    <div className="ocr-step-label">Detected Text Regions</div>
                    <img 
                      src={`${API_URL}${step.highlighted_image}`} 
                      alt="Highlighted text"
                      className="ocr-step-image"
                    />
                  </div>
                )}
                {step.detected_text && (
                  <div className="ocr-step-detected-text">
                    <strong>Detected:</strong> {step.detected_text}
                  </div>
                )}
                {step.text && (
                  <div className="ocr-step-detected-text">
                    <strong>Raw Text:</strong> {step.text}
                  </div>
                )}
                {step.original && step.cleaned && (
                  <div className="ocr-step-cleaning">
                    <div><strong>Original:</strong> {step.original}</div>
                    <div><strong>Cleaned:</strong> {step.cleaned}</div>
                  </div>
                )}
                {step.final_text && (
                  <div className="ocr-step-final-text">
                    <strong>Final Result:</strong> {step.final_text}
                  </div>
                )}
                {step.reason && (
                  <div className="ocr-step-reason">{step.reason}</div>
                )}
                {index < result.ocr_processing_steps.length - 1 && (
                  <div className="ocr-step-arrow">↓</div>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Document Requirements Section */}
      {result.document_provided && result.document_requirements && (
        <div className="document-requirements-section">
          <h3>{result.document_name || 'Regulation Requirements'}</h3>
          
          {result.document_requirements.ppe_requirements && 
           result.document_requirements.ppe_requirements.length > 0 && (
            <div className="requirement-group">
              <h4>PPE Requirements</h4>
              <ul>
                {result.document_requirements.ppe_requirements.map((req, i) => (
                  <li key={i}>{req}</li>
                ))}
              </ul>
            </div>
          )}

          {result.document_requirements.access_requirements && 
           result.document_requirements.access_requirements.length > 0 && (
            <div className="requirement-group">
              <h4>Access & Protection</h4>
              <ul>
                {result.document_requirements.access_requirements.map((req, i) => (
                  <li key={i}>{req}</li>
                ))}
              </ul>
            </div>
          )}

          {result.document_requirements.personnel_limits && 
           Object.keys(result.document_requirements.personnel_limits).length > 0 && (
            <div className="requirement-group">
              <h4>Personnel Limits</h4>
              <pre>{JSON.stringify(result.document_requirements.personnel_limits, null, 2)}</pre>
            </div>
          )}

          {result.document_requirements.inspection_requirements && 
           result.document_requirements.inspection_requirements.length > 0 && (
            <div className="requirement-group">
              <h4>Inspection Requirements</h4>
              <ul>
                {result.document_requirements.inspection_requirements.map((req, i) => (
                  <li key={i}>{req}</li>
                ))}
              </ul>
            </div>
          )}

          {result.document_requirements.equipment_requirements && 
           result.document_requirements.equipment_requirements.length > 0 && (
            <div className="requirement-group">
              <h4>Equipment Requirements</h4>
              <ul>
                {result.document_requirements.equipment_requirements.map((req, i) => (
                  <li key={i}>{req}</li>
                ))}
              </ul>
            </div>
          )}

          {result.document_requirements.zone_requirements && 
           result.document_requirements.zone_requirements.length > 0 && (
            <div className="requirement-group">
              <h4>Zone Requirements</h4>
              <ul>
                {result.document_requirements.zone_requirements.map((req, i) => (
                  <li key={i}>{req}</li>
                ))}
              </ul>
            </div>
          )}

          {result.document_requirements.other_requirements && 
           result.document_requirements.other_requirements.length > 0 && (
            <div className="requirement-group">
              <h4>Other Requirements</h4>
              <ul>
                {result.document_requirements.other_requirements.map((req, i) => (
                  <li key={i}>{req}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Cross-Check Results */}
      {result.document_provided && result.cross_check && (
        <div className="cross-check-section">
          <h3>Document Cross-Check Analysis {result.document_name && `(${result.document_name} selected from dropdown; CDM 2015 and HSE Guidelines automatically included as general UK construction regulations)`}</h3>

          {/* Regulatory Compliance Status */}
          <div className="regulatory-status">
            <div className={`status-badge ${result.cross_check.regulatory_checks.cdm_2015_compliant ? 'compliant' : 'non-compliant'}`}>
              {result.cross_check.regulatory_checks.cdm_2015_compliant ? '✓' : '✗'} CDM 2015
            </div>
            <div className={`status-badge ${result.cross_check.regulatory_checks.hse_compliant ? 'compliant' : 'non-compliant'}`}>
              {result.cross_check.regulatory_checks.hse_compliant ? '✓' : '✗'} HSE Guidelines
            </div>
          </div>

          {/* Cross-Check Violations */}
          {result.cross_check.violations && result.cross_check.violations.length > 0 && (
            <div className="cross-check-violations">
              <h4>🚨 Document vs Photo Violations</h4>
              {result.cross_check.violations.map((violation, index) => (
                <div key={index} className={`violation-card severity-${violation.severity}`}>
                  <div className="violation-header">
                    <span className="severity-badge">{violation.severity.toUpperCase()}</span>
                    <span className="regulation-badge">{violation.regulation}</span>
                  </div>
                  <div className="violation-body">
                    <p><strong>Required:</strong> {violation.requirement}</p>
                    <p><strong>Photo Evidence:</strong> {violation.photo_evidence}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Compliant Items */}
          {result.cross_check.compliant && result.cross_check.compliant.length > 0 && (
            <div className="cross-check-compliant">
              <h4>✓ Compliant Items</h4>
              <ul>
                {result.cross_check.compliant.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {result.cross_check.warnings && result.cross_check.warnings.length > 0 && (
            <div className="cross-check-warnings">
              <h4>⚠️ Manual Verification Required</h4>
              {result.cross_check.warnings.map((warning, index) => (
                <div key={index} className="warning-card">
                  <p><strong>{warning.issue}</strong></p>
                  <p className="warning-reason">{warning.reason}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

