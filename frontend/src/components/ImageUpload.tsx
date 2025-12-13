import { useRef, useState, useEffect } from 'react'

interface ImageUploadProps {
  onUpload: (imageFile: File, documentId?: string) => void
}

interface Document {
  id: string
  name: string
  description: string
  official_link?: string
}

export default function ImageUpload({ onUpload }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [selectedDocument, setSelectedDocument] = useState<string>('')
  const [documents, setDocuments] = useState<Document[]>([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const imageInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/documents')
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      } else {
        console.error('Failed to fetch documents:', response.status, response.statusText)
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err)
      // Set empty array on error so dropdown still shows "None" option
      setDocuments([])
    } finally {
      setLoadingDocs(false)
    }
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageFile(e.dataTransfer.files[0])
    }
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0])
    }
  }

  const handleImageFile = (file: File) => {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']
    if (!validTypes.includes(file.type)) {
      alert('Please upload a PNG, JPG, or WEBP image')
      return
    }

    if (file.size > 16 * 1024 * 1024) {
      alert('File size must be less than 16MB')
      return
    }

    setImageFile(file)
    
    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleSubmit = () => {
    if (!imageFile) {
      alert('Please select an image first')
      return
    }

    onUpload(imageFile, selectedDocument || undefined)
  }

  const handleRemoveImage = () => {
    setImageFile(null)
    setImagePreview(null)
    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }

  return (
    <div className="upload-section">
      {/* Image Upload */}
      <div
        className={`upload-area ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={imageInputRef}
          type="file"
          className="upload-input"
          accept="image/png,image/jpeg,image/jpg,image/webp"
          onChange={handleImageChange}
        />

        <div className="upload-content">
          <svg
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>

          <h2>Upload Construction Site Photo</h2>
          <p>Required - Photo of construction site to analyze</p>
          {imageFile && (
            <p className="file-selected">✓ Selected: {imageFile.name}</p>
          )}

          {!imagePreview && (
            <button 
              onClick={() => imageInputRef.current?.click()} 
              className="btn-primary"
            >
              {imageFile ? 'Change Image' : 'Select Image'}
            </button>
          )}
        </div>
      </div>

      {/* Image Preview */}
      {imagePreview && (
        <div className="image-preview-container">
          <div className="image-preview-header">
            <h3>Image Preview</h3>
            <button
              onClick={handleRemoveImage}
              className="btn-remove-image"
              title="Remove image"
            >
              ×
            </button>
          </div>
          <div className="image-preview-wrapper">
            <img
              src={imagePreview}
              alt="Preview"
              className="image-preview"
            />
          </div>
          <div className="image-preview-actions">
            <button 
              onClick={() => imageInputRef.current?.click()} 
              className="btn-secondary"
            >
              Change Image
            </button>
          </div>
        </div>
      )}

      {/* Document Selection - Optional */}
      <div className="document-selection-area">
        <div className="document-selection-content">
          <svg
            width="48"
            height="48"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
          </svg>

          <div className="document-selection-text">
            <h3>UK Construction Regulation (Optional)</h3>
            <p>Select a regulation to cross-check photo against documented requirements</p>
            
            {loadingDocs ? (
              <p className="loading-docs">Loading regulations...</p>
            ) : documents.length > 0 ? (
              <>
                <select
                  value={selectedDocument}
                  onChange={(e) => setSelectedDocument(e.target.value)}
                  className="document-select"
                >
                  <option value="">None - Photo Only Analysis</option>
                  {documents.map((doc) => (
                    <option key={doc.id} value={doc.id}>
                      {doc.name}
                    </option>
                  ))}
                </select>
              </>
            ) : (
              <p className="loading-docs error-message">
                ⚠️ Unable to load regulations. Please refresh the page.
              </p>
            )}

            {selectedDocument && (
              <div className="document-selected-info">
                <p className="document-description">
                  {documents.find(d => d.id === selectedDocument)?.description}
                </p>
                <div className="document-actions">
                  <button
                    type="button"
                    onClick={() => {
                      const doc = documents.find(d => d.id === selectedDocument)
                      if (doc?.official_link) {
                        window.open(doc.official_link, '_blank', 'noopener,noreferrer')
                      }
                    }}
                    className="btn-view-document"
                    disabled={!documents.find(d => d.id === selectedDocument)?.official_link}
                  >
                    📄 View Official Regulation
                  </button>
                  <details className="document-requirements-preview">
                    <summary>View Requirements Checklist</summary>
                    <DocumentRequirementsPreview documentId={selectedDocument} />
                  </details>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <div className="submit-section">
        <button 
          onClick={handleSubmit} 
          className="btn-analyze"
          disabled={!imageFile}
        >
          Analyze {selectedDocument ? 'with Regulation Cross-Check' : 'Photo Only'}
        </button>
        
        {!selectedDocument && (
          <p className="hint">
            💡 Select a UK regulation for enhanced compliance checking
          </p>
        )}
      </div>

      {/* Info Grid */}
      <div className="info-grid">
        <div className="info-card">
          <h3>Detection</h3>
          <p>YOLOv8 detects workers and equipment</p>
        </div>
        <div className="info-card">
          <h3>OCR</h3>
          <p>Tesseract reads signage text</p>
        </div>
        <div className="info-card">
          <h3>Cross-Check</h3>
          <p>AI compares photo vs UK regulations</p>
        </div>
      </div>
    </div>
  )
}

// Component to show requirements preview
function DocumentRequirementsPreview({ documentId }: { documentId: string }) {
  const [requirements, setRequirements] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchRequirements = async () => {
      try {
        // We'll get this from the backend when document is selected
        // For now, we can fetch it from the analyze endpoint or create a new endpoint
        const response = await fetch(`http://localhost:5001/api/documents/${documentId}/requirements`)
        if (response.ok) {
          const data = await response.json()
          setRequirements(data.requirements)
        }
      } catch (err) {
        console.error('Failed to fetch requirements:', err)
      } finally {
        setLoading(false)
      }
    }

    if (documentId) {
      fetchRequirements()
    }
  }, [documentId])

  if (loading) {
    return <p className="loading-requirements">Loading requirements...</p>
  }

  if (!requirements) {
    return <p className="no-requirements">Requirements not available</p>
  }

  return (
    <div className="requirements-preview">
      {requirements.ppe_requirements && requirements.ppe_requirements.length > 0 && (
        <div className="requirement-category">
          <h4>PPE Requirements</h4>
          <ul>
            {requirements.ppe_requirements.map((req: string, i: number) => (
              <li key={i}>{req}</li>
            ))}
          </ul>
        </div>
      )}

      {requirements.access_requirements && requirements.access_requirements.length > 0 && (
        <div className="requirement-category">
          <h4>Access & Protection</h4>
          <ul>
            {requirements.access_requirements.map((req: string, i: number) => (
              <li key={i}>{req}</li>
            ))}
          </ul>
        </div>
      )}

      {requirements.equipment_requirements && requirements.equipment_requirements.length > 0 && (
        <div className="requirement-category">
          <h4>Equipment Requirements</h4>
          <ul>
            {requirements.equipment_requirements.map((req: string, i: number) => (
              <li key={i}>{req}</li>
            ))}
          </ul>
        </div>
      )}

      {requirements.inspection_requirements && requirements.inspection_requirements.length > 0 && (
        <div className="requirement-category">
          <h4>Inspection Requirements</h4>
          <ul>
            {requirements.inspection_requirements.map((req: string, i: number) => (
              <li key={i}>{req}</li>
            ))}
          </ul>
        </div>
      )}

      {requirements.zone_requirements && requirements.zone_requirements.length > 0 && (
        <div className="requirement-category">
          <h4>Zone Requirements</h4>
          <ul>
            {requirements.zone_requirements.map((req: string, i: number) => (
              <li key={i}>{req}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
