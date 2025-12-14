export default function Header() {
  return (
    <header className="header">
      <div className="container">
        <h1>
          Construction H&S Compliance Analyzer
          <span style={{display: 'block', fontSize: '0.6em', fontWeight: 'normal', marginTop: '0.5em'}}>
            Computer Vision & Document Cross-Check
          </span>
        </h1>
        <p className="subtitle">
          Automated detection of workers and safety signage using computer vision
        </p>
        <blockquote className="quote">
          <p>"Every business uses their <strong>own system</strong> for <strong>compliance & health and safety</strong> so the <strong>supply chain</strong> has to do something <strong>different on every job</strong> — that leads to <strong>tens if not hundreds of different systems</strong> for an <strong>SME</strong> to learn and do."</p>
          <p>"There is <strong>no increase or improvement in productivity</strong> as every time you have to <strong>do something different</strong> there is a <strong>drag on your time</strong>."</p>
          <p>"The <strong>cost becomes ridiculous</strong> as companies have to <strong>buy a different system</strong>, <strong>train people on each system</strong> and <strong>time = money</strong>"</p>
          <cite>
            — <a href="https://www.linkedin.com/in/suzannah-nichol-obe-💙💛-48540a47/" target="_blank" rel="noopener noreferrer" style={{color: 'inherit', textDecoration: 'none'}}>
              Suzannah Nichol OBE, Chief Executive at Build UK
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" style={{display: 'inline-block', marginLeft: '4px', verticalAlign: 'middle'}}>
                <path d="M10 2L2 10M10 2H6M10 2V6M2 10H6M2 10V6" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </a>
          </cite>
        </blockquote>
      </div>
    </header>
  )
}

