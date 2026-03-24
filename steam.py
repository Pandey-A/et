import streamlit as st
import requests
import json
from datetime import datetime, date

# Configuration
API_BASE_URL = "http://127.0.0.1:5050"

st.set_page_config(page_title="Attendance System Tester", layout="wide")
st.title("🧪 Attendance System API Tester")

# Sidebar for API URL configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    st.markdown("---")
    st.info("📝 Test all your Flask API endpoints from this dashboard")

# Create tabs for different endpoints
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👥 View Employees", 
    "➕ Register Employee", 
    "✅ Record Attendance",
    "📊 View Attendance",
    "📄 Generate Report"
])

# TAB 1: View Employees
with tab1:
    st.header("👥 View All Employees")
    
    if st.button("Fetch Employees", key="fetch_emp"):
        with st.spinner("Fetching employees..."):
            try:
                response = requests.get(f"{api_url}/employees")
                if response.status_code == 200:
                    data = response.json()
                    employees = data.get("employees", [])
                    
                    if employees:
                        st.success(f"✅ Found {len(employees)} employees")
                        st.json(employees)
                        
                        # Display in a nice table
                        st.subheader("Employee List")
                        for emp in employees:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("User ID", emp.get("user_id"))
                            with col2:
                                st.metric("Name", emp.get("name"))
                            with col3:
                                st.metric("Role", emp.get("role"))
                            st.markdown("---")
                    else:
                        st.warning("No employees found")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# TAB 2: Register Employee
with tab2:
    st.header("➕ Register New Employee")
    
    with st.form("register_form"):
        st.subheader("Admin Credentials")
        admin_id = st.text_input("Admin User ID", placeholder="admin123")
        
        st.subheader("New Employee Details")
        new_user_id = st.text_input("Employee User ID", placeholder="emp001")
        new_name = st.text_input("Employee Name", placeholder="John Doe")
        new_role = st.selectbox("Role", ["employee", "admin", "manager"])
        
        submitted = st.form_submit_button("Register Employee")
        
        if submitted:
            if not admin_id or not new_user_id or not new_name:
                st.error("❌ All fields are required!")
            else:
                payload = {
                    "added_by": admin_id,
                    "user_id": new_user_id,
                    "name": new_name,
                    "role": new_role
                }
                
                with st.spinner("Registering employee..."):
                    try:
                        response = requests.post(
                            f"{api_url}/employees/register",
                            json=payload
                        )
                        
                        if response.status_code == 201:
                            st.success("✅ Employee registered successfully!")
                            st.json(response.json())
                        elif response.status_code == 403:
                            st.error("❌ Only admin can add new employees")
                        elif response.status_code == 400:
                            st.error("❌ User ID already exists")
                        else:
                            st.error(f"❌ Error {response.status_code}: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")

# TAB 3: Record Attendance
with tab3:
    st.header("✅ Record Attendance")
    
    with st.form("attendance_form"):
        user_id = st.text_input("Employee User ID", placeholder="emp001")
        gesture = st.selectbox("Action", ["check-in", "check-out", "break-start", "break-end"])
        
        submitted = st.form_submit_button("Record Attendance")
        
        if submitted:
            if not user_id:
                st.error("❌ User ID is required!")
            else:
                payload = {
                    "user_id": user_id,
                    "gesture": gesture
                }
                
                with st.spinner("Recording attendance..."):
                    try:
                        response = requests.post(
                            f"{api_url}/attendance/record",
                            json=payload
                        )
                        
                        if response.status_code == 201:
                            st.success("✅ Attendance recorded successfully!")
                            st.json(response.json())
                        elif response.status_code == 404:
                            st.error("❌ Employee not found")
                        else:
                            st.error(f"❌ Error {response.status_code}: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")

# TAB 4: View Attendance
with tab4:
    st.header("📊 View Attendance Records")
    
    col1, col2 = st.columns(2)
    with col1:
        use_filter = st.checkbox("Filter by date range")
    
    start_date = None
    end_date = None
    
    if use_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today())
        with col2:
            end_date = st.date_input("End Date", value=date.today())
    
    if st.button("Fetch Attendance", key="fetch_att"):
        with st.spinner("Fetching attendance records..."):
            try:
                params = {}
                if use_filter and start_date and end_date:
                    params = {
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d")
                    }
                
                response = requests.get(
                    f"{api_url}/attendance",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("attendance", [])
                    
                    if records:
                        st.success(f"✅ Found {len(records)} attendance records")
                        
                        # Display records
                        for idx, record in enumerate(records, 1):
                            with st.expander(f"Record {idx}: {record.get('user_id')} - {record.get('gesture')}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write("**User ID:**", record.get("user_id"))
                                with col2:
                                    st.write("**Gesture:**", record.get("gesture"))
                                with col3:
                                    st.write("**Timestamp:**", record.get("timestamp"))
                        
                        # Show raw JSON
                        with st.expander("📄 View Raw JSON"):
                            st.json(records)
                    else:
                        st.warning("No attendance records found")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# TAB 5: Generate PDF Report
with tab5:
    st.header("📄 Generate & Send PDF Report")
    
    st.info("📤 This will generate a PDF report and send it to Telegram")
    
    with st.form("report_form"):
        use_date_range = st.checkbox("Filter by date range", value=True)
        
        report_start = None
        report_end = None
        
        if use_date_range:
            col1, col2 = st.columns(2)
            with col1:
                report_start = st.date_input("Start Date", value=date.today(), key="report_start")
            with col2:
                report_end = st.date_input("End Date", value=date.today(), key="report_end")
        
        submitted = st.form_submit_button("Generate & Send Report")
        
        if submitted:
            payload = {}
            if use_date_range and report_start and report_end:
                payload = {
                    "start_date": report_start.strftime("%Y-%m-%d"),
                    "end_date": report_end.strftime("%Y-%m-%d")
                }
            
            with st.spinner("Generating report and sending to Telegram..."):
                try:
                    response = requests.post(
                        f"{api_url}/send_report_pdf",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ PDF report generated and sent to Telegram successfully!")
                        st.balloons()
                        st.json(response.json())
                    elif response.status_code == 404:
                        st.warning("⚠️ No records found for the given date range")
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💡 Make sure your Flask API is running on <code>localhost:5001</code></p>
    <p>Test all endpoints and verify responses in real-time!</p>
</div>
""", unsafe_allow_html=True)