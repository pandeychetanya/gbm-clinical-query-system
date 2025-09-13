# Enhanced Adjuvant Therapy for Newly Diagnosed Glioblastoma With Partial Surgical Resection or Short-term Progression: Bayesian Adaptive Randomized Phase II Study

**Principal Investigator**: Zhang Ting  
**Institution**: Second Affiliated Hospital of Zhejiang University  
**Protocol Version**: 1.0  
**Date**: April 12, 2025  
**Study Type**: Prospective, Randomized, Controlled, Open-label, Single-center Phase II Study  
**Extracted**: September 2024

---

## STUDY OVERVIEW

### Primary Objective
Evaluate the effect of intensified adjuvant therapy on 3-month, 6-month, and 12-month progression-free survival (PFS) rates in newly diagnosed GBM patients undergoing partial surgical resection or short-term recurrence progression, using RANO 2.0 criteria.

### Secondary Objectives
1. Evaluate impact on 1-year and 2-year overall survival rates and quality of life
2. Assess safety and incidence of radiation necrosis
3. Compare efficacy across four treatment arms using Bayesian adaptive methodology

### Clinical Rationale
**Unmet Medical Need**: Patients with partial surgical resection or short-term recurrence have worse prognosis than those with complete resection. Current evidence shows:
- 100% resection: 16 months OS
- 90% resection: 13.8 months OS  
- 80% resection: 12.8 months OS
- 78% resection: 12.5 months OS

**Innovation**: First Bayesian adaptive randomized trial for GBM intensified adjuvant therapy, allowing dynamic allocation based on real-time efficacy data.

---

## TREATMENT ARMS AND PROTOCOLS

### Four-Arm Design (1:1:1:1 Initial Randomization)

#### ARM A: Standard Stupp Protocol
**Concomitant Phase** (2-6 weeks post-surgery):
- **Radiotherapy**: PTV1 60Gy/30F (high-risk), 54Gy/30F (low-risk)
- **TMZ**: 75 mg/m² PO daily during RT
- **Duration**: 42-49 consecutive days with focal radiotherapy

**Maintenance Phase** (28 days after RT completion):
- **Cycle 1**: 150 mg/m² PO daily × 5 days (28-day cycles)
- **Cycles 2-6**: 200 mg/m² PO daily × 5 days if no toxicity in Cycle 1

#### ARM B: Stupp + PD-1/VEGF Dual Antibody
**Concomitant Phase**: Same as Arm A

**Enhanced Maintenance Phase**:
- **TMZ**: Standard dosing as Arm A
- **PD-1/VEGF dual antibody**: 20 mg/kg IV infusion
- **Cycle**: Every 21 days
- **Duration**: Until progression or unacceptable toxicity

#### ARM C: Stupp + PD-1/CTLA-4 Dual Antibody  
**Concomitant Phase**: Same as Arm A

**Enhanced Maintenance Phase**:
- **TMZ**: Standard dosing as Arm A
- **PD-1/CTLA-4 dual antibody**: 6 mg/kg IV infusion
- **Cycle**: Every 14 days
- **Duration**: Until progression or unacceptable toxicity

#### ARM D: Modified Stupp with Dose-Escalated Radiation
**Enhanced Concomitant Phase**:
- **PGTV**: 66Gy/30F to residual/recurrent lesions (dose escalation)
- **PTV1**: 60Gy/30F (high-risk), 54Gy/30F (low-risk)
- **TMZ**: 75 mg/m² PO daily during RT

**Maintenance Phase**: Standard 6-cycle TMZ as Arm A

---

## BAYESIAN ADAPTIVE RANDOMIZATION METHODOLOGY

### Novel Statistical Design Features

#### Stage 1: Initial Randomization (First 28 Patients)
- **Equal allocation**: 7 patients per arm (1:1:1:1 ratio)
- **Purpose**: Establish baseline safety and preliminary efficacy data
- **Duration**: First 3-4 months of enrollment

#### Stage 2: Adaptive Randomization (Patients 29+)
**Re-evaluation Intervals**: Every 16 patients enrolled
**Primary Endpoint Assessment**: 12-month PFS rate estimation
**Dynamic Allocation**: Bayesian posterior probability calculations
**Software**: Professional randomization using SAS or R packages

#### Adaptive Algorithm Specifications
**Prior Knowledge Integration**:
- Historical Stupp regimen: 40% 12-month PFS (institutional data)
- Expected experimental arms: 70% 12-month PFS (target improvement)

**Posterior Updates**:
- Beta-binomial conjugate priors
- Continuous probability estimation refinement
- Dynamic randomization ratio adjustment based on accumulating efficacy data

**Statistical Power**:
- Alpha error: 0.05
- Beta error: 0.20
- Power: 80% to detect PFS difference between arms

---

## PATIENT POPULATION AND ELIGIBILITY

### Inclusion Criteria (Key Requirements)
1. **Age**: ≥18 years old
2. **Diagnosis**: Pathologically confirmed glioblastoma (WHO Grade 4)
3. **Surgical Status**: Partial surgical resection OR short-term recurrence/progression 2-6 weeks post-surgery (before RT)
4. **Performance Status**: Adequate organ function
5. **Hematologic Function**:
   - ANC ≥1.5 × 10⁹/L
   - Platelets ≥75 × 10⁹/L
   - Hemoglobin ≥9 g/dL
6. **Hepatic Function**:
   - Bilirubin ≤1.5× ULN
   - AST/ALT ≤1.5× ULN
7. **Renal Function**: Creatinine ≤1.5× ULN
8. **Cardiac Function**: LVEF ≥50%

### Major Exclusion Criteria
1. **Prior Treatments**: Previous brain radiation therapy
2. **Concurrent Malignancies**: Other active cancers within 3 years
3. **Medical Comorbidities**: 
   - Active infections requiring systemic treatment
   - Uncontrolled cardiovascular disease (NYHA Class III-IV)
   - QTcF >480 milliseconds
4. **Pregnancy/Lactation**: Excluded with mandatory contraception requirements

### Target Sample Size
**Total Enrollment**: 210 patients (accounting for 5% dropout)
**Effective Analysis**: 200 patients (50 per arm)
**Enrollment Period**: 24 months
**Follow-up Period**: 24 months

---

## ENDPOINT ASSESSMENTS AND MONITORING

### Primary Efficacy Endpoints
1. **3-month PFS rate** (RANO 2.0 criteria)
2. **6-month PFS rate** (RANO 2.0 criteria)
3. **12-month PFS rate** (RANO 2.0 criteria)

### Secondary Efficacy Endpoints
1. **Overall Survival**: 1-year and 2-year rates
2. **Quality of Life**: Changes relative to baseline and time to deterioration
3. **Comparative Efficacy**: Between-arm differences in PFS/OS

### Safety Endpoints
1. **Adverse Events**: NCI-CTCAE 5.0 grading
2. **Radiation Necrosis**: Incidence and severity assessment
3. **Treatment-Related Mortality**: 90-day safety follow-up

### Imaging Protocol
**Schedule**: 
- Baseline, post-operative, pre-RT, post-RT
- Monthly MRI enhanced scans during follow-up
- PFS assessment until progression or study completion

---

## SAFETY MONITORING AND ADVERSE EVENT MANAGEMENT

### Adverse Event Definitions
**AE Period**: From informed consent signature until 90 days post-last treatment
**SAE Criteria**: Death, life-threatening events, hospitalization, disability, congenital anomalies

### Grading and Causality Assessment
**Severity**: NCI-CTCAE 5.0 criteria
**Causality Levels**:
1. Definitely related
2. Possibly related  
3. Possibly unrelated
4. Definitely unrelated
5. Unable to determine

### Reporting Timeline
**SAE Reporting**: Within 24 hours to:
- Ethics committee
- Regulatory authorities (NMPA)
- Health administrative departments
- Study sponsor

### Treatment Discontinuation Criteria
1. **Patient/Legal Representative Request**
2. **Protocol-Specified Adverse Events**
3. **Concurrent Disease Development**
4. **Investigator Decision**
5. **Pregnancy**
6. **Poor Compliance**
7. **Treatment Completion**

---

## DATA MANAGEMENT AND QUALITY ASSURANCE

### Case Report Form (CRF) Management
**Format**: Triplicate carbonless copies
**Data Entry**: Duplicate entry with computer validation
**Quality Control**: 
- Real-time data monitoring
- Query resolution processes
- Database lock procedures

### Regulatory Compliance
**Document Retention**: Hospital and national regulations
**Training Requirements**: All investigators on SOPs and regulations
**Laboratory QC**: Ministry of Health clinical laboratory center participation

### Statistical Analysis Plan
**Interim Analyses**: Every 16 patients for adaptive randomization
**Final Analysis**: Intention-to-treat and per-protocol populations
**Bayesian Methods**: Posterior probability calculations for treatment selection

---

## CLINICAL IMPLICATIONS AND SIGNIFICANCE

### Addressing Critical Unmet Need
**High-Risk Population**: Patients with partial resection or early progression represent ~30-40% of newly diagnosed GBM cases with particularly poor prognosis

### Innovation in Clinical Trial Design
**Bayesian Adaptive Methodology**:
- Increases trial efficiency
- Reduces patient exposure to inferior treatments  
- Enables real-time learning from accumulating data
- Provides framework for future GBM trial optimization

### Treatment Strategy Evaluation
**Four Distinct Approaches**:
1. **Standard care validation** (Arm A)
2. **Immunotherapy enhancement** (Arms B & C with dual checkpoint/angiogenesis inhibition)
3. **Radiation dose escalation** (Arm D with focal boost to residual disease)

### Expected Impact on Clinical Practice
**If Positive Results**:
- New treatment paradigm for high-risk GBM patients
- Evidence-based approach to treatment intensification
- Potential integration into national treatment guidelines
- Foundation for Phase III confirmatory trials

### Regulatory Pathway
**Study Classification**: Investigational New Drug (IND) status required for combination therapies
**Approval Strategy**: Conditional approval pathway based on PFS improvement in high-risk population

---

## STUDY TIMELINE AND MILESTONES

### Phase 1: Study Initiation (Months 1-3)
- Ethics committee approval
- Regulatory submissions
- Site activation and staff training
- First patient enrollment

### Phase 2: Enrollment Period (Months 1-24)
- Patient accrual: 210 patients over 24 months
- First interim analysis: After 44 patients (Month 6)
- Subsequent interim analyses: Every 16 patients
- Safety monitoring: Continuous

### Phase 3: Follow-up Period (Months 13-48)
- Primary endpoint assessment: 12-month PFS
- Long-term survival follow-up: 24 months minimum  
- Quality of life assessments: Quarterly
- Safety follow-up: 90 days post-treatment

### Phase 4: Analysis and Reporting (Months 25-50)
- Database lock and cleaning
- Statistical analysis execution
- Clinical study report preparation
- Regulatory submissions and publication

---

*This protocol represents the first Bayesian adaptive randomized Phase II study specifically designed for newly diagnosed GBM patients with partial surgical resection or short-term progression, addressing a critical unmet medical need through innovative trial methodology and intensified adjuvant therapy approaches.*