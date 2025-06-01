# Solar Flare Analysis ML Application

A production-ready machine learning application for analyzing GOES/EXIS satellite data to detect, separate, and analyze solar flares with special focus on nanoflare identification.

## 🌟 Features

- **ML-Powered Flare Detection**: Advanced machine learning algorithms to separate overlapping solar flares
- **Nanoflare Identification**: Automatic detection of nanoflares with |α| > 2 for corona heating studies
- **Interactive Dashboard**: Modern React-based frontend with real-time visualization
- **Statistical Analysis**: Comprehensive power-law analysis and energy distribution studies
- **Production Ready**: Full-stack deployment with Node.js backend and Python ML integration

## 🚀 Technologies

### Frontend
- **Next.js 15** - React framework with App Router
- **React 19** - Modern UI components with concurrent features
- **Tailwind CSS 4** - Utility-first CSS framework
- **Recharts** - Interactive data visualization
- **TypeScript** - Type-safe development

### Backend
- **Python Flask** - ML model serving API
- **TensorFlow/Scikit-learn** - Machine learning frameworks
- **NetCDF4** - GOES satellite data processing
- **NumPy/SciPy** - Scientific computing

## 📋 Prerequisites

- **Node.js** 18+ 
- **Python** 3.8+
- **Git**
- **Virtual Environment** (recommended: `goesflareenv`)

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\ml_app
```

### 2. Install Dependencies

**Frontend (Node.js):**
```bash
npm install
```

**Backend (Python):**
```bash
# Activate your virtual environment first
pip install flask flask-cors tensorflow scikit-learn numpy scipy netcdf4 matplotlib
```

### 3. Environment Configuration

Create `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:3000/api
PYTHON_API_URL=http://localhost:5000
```

## 🚀 Running the Application

### Option 1: Automated Startup (Recommended)

**Windows PowerShell:**
```powershell
.\start_app.ps1
```

This script will:
- Check system requirements
- Start Python ML backend on port 5000
- Start Next.js frontend on port 3000
- Handle graceful shutdown

### Option 2: Manual Startup

**Terminal 1 - Python Backend:**
```bash
python python_bridge.py --host 0.0.0.0 --port 5000
```

**Terminal 2 - Next.js Frontend:**
```bash
npm run dev
```

## 🌐 Application URLs

- **Frontend Dashboard**: http://localhost:3000
- **Python API Backend**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## 📊 Usage

### 1. Upload GOES Data
- Click "Choose file" to select a GOES XRS NetCDF (.nc) file
- Supported formats: GOES-16/17/18 XRS data

### 2. Run Analysis
- Click "Run ML Analysis" to process the data
- Progress bar shows real-time analysis status
- Notifications provide feedback on each step

### 3. View Results
- **Flare Timeline**: Time series visualization of detected flares
- **Energy Distribution**: Histogram of flare energy ranges
- **Power Law Analysis**: Scatter plot showing α vs Energy relationship
- **Nanoflare Table**: Detailed list of identified nanoflares (|α| > 2)

### 4. Export Data
- Click "Export Results" to download analysis in JSON format
- Includes all detected flares, statistics, and metadata

## 🔬 ML Model Details

### Flare Decomposition Model
- **Algorithm**: Custom neural network with time-series analysis
- **Input**: GOES XRS flux data (1-8 Å and 0.5-4 Å channels)
- **Output**: Separated individual flare components
- **Training**: Synthetic flare data with known parameters

### Nanoflare Detection
- **Criteria**: Power-law index |α| > 2
- **Purpose**: Corona heating mechanism studies
- **Validation**: Statistical significance testing

### Power Law Analysis
- **Formula**: N(E) ∝ E^(-α)
- **Range**: 10^26 - 10^32 Joules
- **Applications**: Energy budget calculations

## 📁 Project Structure

```
ml_app/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── api/               # API routes
│   │   │   ├── analyze/       # Analysis endpoint
│   │   │   └── model/         # Model management
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Main dashboard
│   └── components/            # React components
│       ├── SolarFlareAnalyzer.tsx
│       └── NotificationSystem.tsx
├── python_bridge.py           # Python ML API server
├── start_app.ps1             # Windows startup script
└── package.json              # Dependencies
```

## 🧪 Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
npm start
```

### Python Model Development
```bash
python -c "from python_bridge import initialize_model; initialize_model()"
```

## 📈 Performance

- **Analysis Speed**: ~30 seconds for typical GOES day file
- **Memory Usage**: <2GB for full analysis
- **Concurrent Users**: Supports multiple simultaneous analyses
- **Data Size**: Optimized for files up to 100MB

## 🔧 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Kill existing processes
npx kill-port 3000
npx kill-port 5000
```

**Python Dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**Virtual Environment Issues:**
```bash
# Create new environment
python -m venv goesflareenv
# Activate and install
goesflareenv\Scripts\activate
pip install -r requirements.txt
```

## 📚 Scientific Background

### Solar Flare Physics
Solar flares are sudden releases of electromagnetic energy in the solar corona, classified by their X-ray flux:
- **A, B, C**: Background levels
- **M**: Medium flares (10^-5 to 10^-4 W/m²)
- **X**: Major flares (>10^-4 W/m²)

### Nanoflares
Tiny flares with energies ~10^24-10^27 erg, potentially responsible for coronal heating through power-law energy distribution with steep indices (α > 2).

### GOES/EXIS Data
The Geostationary Operational Environmental Satellites carry the Extreme Ultraviolet and X-ray Irradiance Sensors (EXIS) providing:
- **XRS**: X-Ray Sensor (0.5-4 Å, 1-8 Å)
- **EUVS**: Extreme UV Sensor
- **1-minute cadence**: High temporal resolution

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA GOES Team for satellite data
- Solar Physics community for scientific guidance
- Open source libraries and frameworks used

## 📞 Support

For support, please open an issue in the repository or contact the development team.

---

**Built with ❤️ for Solar Physics Research**
