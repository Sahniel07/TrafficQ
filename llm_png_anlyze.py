import os
import glob
from dashscope import MultiModalConversation
import base64
from datetime import datetime
import json

# Set API key directly in the code
import dashscope
dashscope.api_key = "sk-fbfbf8f489a043ac86e6ec65586c34dd"
# Set international API URL
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def format_vehicle_data(vehicle_id, vehicle_data):
    """
    Convert vehicle JSON data to a formatted text report
    
    Parameters:
    vehicle_id (str): Vehicle ID
    vehicle_data (dict): Vehicle data dictionary
    
    Returns:
    str: Formatted text report
    """
    if not vehicle_data:
        return f"No data found for vehicle ID: {vehicle_id}"
    
    report = []
    report.append(f"*****  ANALYSIS REPORT FOR VEHICLE: {vehicle_id}  ******")
    report.append("=" * 80)
    report.append("")
    
    # Add turn analysis
    if "final_turn" in vehicle_data:
        turn_data = vehicle_data["final_turn"]
        report.append("📍 FINAL TURN ANALYSIS")
        if turn_data.get("status") == "success":
            initial = turn_data.get("initial_direction", {})
            final = turn_data.get("final_direction", {})
            net_turn = turn_data.get("net_turn", {})
            
            report.append(f"   • Initial direction: {initial.get('degrees', 'N/A')}° {initial.get('arrow', '')} at {initial.get('formatted_time', 'N/A')}")
            report.append(f"   • Final direction:   {final.get('degrees', 'N/A')}° {final.get('arrow', '')} at {final.get('formatted_time', 'N/A')}")
            report.append(f"   • Net turn:          {net_turn.get('degrees', 'N/A')}° {net_turn.get('icon', '')} ({net_turn.get('type', 'N/A')})")
        else:
            report.append("   • Analysis failed")
        report.append("")
    
    # Add speed statistics
    if "speed_stats" in vehicle_data:
        speed_data = vehicle_data["speed_stats"]
        report.append("🚀 SPEED STATISTICS")
        if speed_data.get("status") == "success":
            overall = speed_data.get("overall_avg", {})
            moving = speed_data.get("moving_avg", {})
            min_speed = speed_data.get("min_speed", {})
            max_speed = speed_data.get("max_speed", {})
            
            report.append(f"   • Overall average:   {overall.get('mps', 'N/A')} m/s ({overall.get('kmh', 'N/A')} km/h)")
            report.append(f"   • Moving average:    {moving.get('mps', 'N/A')} m/s ({moving.get('kmh', 'N/A')} km/h)")
            report.append(f"   • Minimum speed:     {min_speed.get('mps', 'N/A')} m/s ({min_speed.get('kmh', 'N/A')} km/h)")
            report.append(f"   • Maximum speed:     {max_speed.get('mps', 'N/A')} m/s ({max_speed.get('kmh', 'N/A')} km/h)")
        else:
            report.append("   • Analysis failed")
        report.append("")
    
    # Add acceleration analysis
    if "acceleration" in vehicle_data:
        accel_data = vehicle_data["acceleration"]
        report.append("⚡ ACCELERATION ANALYSIS")
        if accel_data.get("status") == "success":
            report.append(f"   • Maximum acceleration:    {accel_data.get('max_acceleration', 'N/A')} m/s²")
            report.append(f"   • Maximum deceleration:    {accel_data.get('max_deceleration', 'N/A')} m/s²")
        else:
            report.append("   • Analysis failed")
        report.append("")
    
    # Add stop events
    if "stop_events" in vehicle_data:
        stop_data = vehicle_data["stop_events"]
        report.append("🛑 STOP EVENTS")
        count = stop_data.get("count", 0)
        events = stop_data.get("events", [])
        
        report.append(f"   • Total stop events: {count}")
        if count > 0:
            for i, event in enumerate(events):
                start = event.get("start_time", "N/A")
                end = event.get("end_time", "N/A")
                duration = event.get("duration", "N/A")
                report.append(f"     └─ Stop #{i+1}: {start} → {end} ({duration} seconds)")
        
        report.append("")
    
    return "\n".join(report)

def analyze_images(folder_path):
    """
    Analyze all PNG images in the folder using Qwen-VL model and generate individual reports
    
    Parameters:
    folder_path (str): Path to the folder containing PNG images
    """
    # Get all PNG images
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    
    if not png_files:
        print(f"No PNG images found in {folder_path}")
        return
    
    print(f"Found {len(png_files)} PNG images, starting analysis...")
    
    # Load output_results_2.json
    output_results_path = "output_results_2.json"
    output_results_data = {}
    if os.path.exists(output_results_path):
        try:
            with open(output_results_path, "r") as f:
                output_results_data = json.load(f)
            print(f"Loaded {output_results_path}")
        except Exception as e:
            print(f"Failed to load {output_results_path}: {e}")
    else:
        print(f"File not found: {output_results_path}")
    
    # Create reports folder
    reports_folder = os.path.join(folder_path, "trajectory_reports")
    os.makedirs(reports_folder, exist_ok=True)
    
    # Create CSS file for better styling
    css_file = os.path.join(reports_folder, "style.css")
    with open(css_file, "w", encoding="utf-8") as f:
        f.write("""
        :root {
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --accent-color: #e74c3c;
            --light-bg: #f8f9fa;
            --dark-bg: #343a40;
            --light-text: #f8f9fa;
            --dark-text: #343a40;
            --box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark-text);
            background-color: var(--light-bg);
            padding: 0;
            margin: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: var(--light-text);
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: var(--box-shadow);
        }
        
        h1, h2, h3 {
            margin-bottom: 15px;
            color: var(--secondary-color);
        }
        
        header h1 {
            color: var(--light-text);
        }
        
        .meta-info {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 20px;
        }
        
        .vehicle-data {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            box-shadow: var(--box-shadow);
            border-left: 5px solid var(--primary-color);
        }
        
        .vehicle-data h2 {
            color: var(--primary-color);
            border-bottom: 1px solid #ccc;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        .image-analysis {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
            box-shadow: var(--box-shadow);
        }
        
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: var(--box-shadow);
            margin: 15px 0;
        }
        
        pre {
            white-space: pre-wrap;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', Courier, monospace;
            overflow-x: auto;
            border-left: 3px solid var(--primary-color);
        }
        
        .analysis-result {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: var(--box-shadow);
        }
        
        .error-box {
            background-color: #ffebee;
            border-left: 5px solid var(--accent-color);
        }
        
        .nav-links {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        
        .nav-links a {
            background-color: var(--primary-color);
            color: white;
            padding: 10px 15px;
            text-decoration: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        
        .nav-links a:hover {
            background-color: var(--secondary-color);
        }
        
        /* Index page styles */
        .report-list {
            list-style-type: none;
        }
        
        .report-item {
            background-color: white;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: var(--box-shadow);
            transition: transform 0.2s;
        }
        
        .report-item:hover {
            transform: translateY(-3px);
        }
        
        .report-item a {
            display: block;
            text-decoration: none;
            color: var(--primary-color);
            font-weight: bold;
        }
        
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            header {
                padding: 15px;
            }
            
            .vehicle-data, .image-analysis {
                padding: 15px;
            }
        }
        """)
    
    # Process each image and generate individual reports
    for index, png_file in enumerate(png_files):
        file_name = os.path.basename(png_file)
        file_base_name = os.path.splitext(file_name)[0]
        print(f"Analyzing ({index+1}/{len(png_files)}): {file_name}")
        
        # Create report file path for current image
        report_filename = f"{file_base_name}_analysis.html"
        report_path = os.path.join(reports_folder, report_filename)
        
        # Look for a JSON file with the same base name
        json_file = os.path.join(folder_path, f"{file_base_name}.json")
        vehicle_id = None
        vehicle_data = None
        vehicle_text_data = ""  # Formatted text data
        all_vehicles_text_data = []  # Text data for all vehicles in a cluster
        
        # Try to extract vehicle_id from JSON file and get related data
        if os.path.exists(json_file):
            try:
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                    print(f"JSON file content: {json.dumps(json_data, ensure_ascii=False)[:200]}...")
                    
                    # Check if this is a cluster trajectory JSON
                    if "cluster_id" in json_data:
                        cluster_id = json_data['cluster_id']
                        print(f"This is a cluster trajectory JSON, cluster ID: {cluster_id}")
                        
                        # Extract vehicle_ids from vehicles array
                        if "vehicles" in json_data and isinstance(json_data["vehicles"], list):
                            vehicles = json_data["vehicles"]
                            vehicle_ids = [vehicle["vehicle_id"] for vehicle in vehicles if "vehicle_id" in vehicle]
                            print(f"Cluster contains {len(vehicle_ids)} vehicle IDs: {vehicle_ids}")
                            
                            # Collect data for all vehicles
                            all_vehicles_data = {}
                            for v_id in vehicle_ids:
                                if v_id in output_results_data:
                                    all_vehicles_data[v_id] = output_results_data[v_id]
                                    print(f"Found data for vehicle ID: {v_id}")
                                    # Generate formatted text data for each vehicle
                                    all_vehicles_text_data.append(format_vehicle_data(v_id, output_results_data[v_id]))
                            
                            if all_vehicles_data:
                                vehicle_data = all_vehicles_data
                                print(f"Retrieved data for {len(all_vehicles_data)} vehicles from output_results_2.json")
                            else:
                                print("No vehicle data found")
                        else:
                            print("No vehicles array found in JSON file or format is incorrect")
                    else:
                        # Try to get vehicle_id directly
                        vehicle_id = json_data.get("vehicle_id")
                        print(f"Retrieved vehicle_id from {json_file}: {vehicle_id}")
                        
                        if vehicle_id and vehicle_id in output_results_data:
                            vehicle_data = output_results_data[vehicle_id]
                            vehicle_text_data = format_vehicle_data(vehicle_id, vehicle_data)
                            print(f"Retrieved data for vehicle_id: {vehicle_id} from output_results_2.json")
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        try:
            # Read image and convert to Base64
            with open(png_file, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare system prompt and user prompt
            system_prompt = 'You are an expert in analyzing vehicle trajectory maps. These images show vehicle trajectory clustering results, where the background is a road map and colored lines represent trajectories of different vehicles.'
            user_prompt = 'Please analyze the vehicle trajectory clustering features shown in this image. Describe the main paths, turning patterns, and characteristics. What type of driving behavior does this represent?'
            
            # Enhance prompt with vehicle data if available
            if vehicle_text_data:
                user_prompt = f'Here is the driving data for the vehicle:\n\n{vehicle_text_data}\n\nPlease analyze the trajectory features shown in this image, combining this data. Describe the main paths, turning patterns, and characteristics. What type of driving behavior does this represent?'
            elif all_vehicles_text_data:
                # Limit text length to avoid overly long prompts
                combined_text = "\n\n".join(all_vehicles_text_data[:3])
                if len(all_vehicles_text_data) > 3:
                    combined_text += f"\n\n... Data for {len(all_vehicles_text_data) - 3} more vehicles not shown ..."
                
                user_prompt = f'Here is the driving data for vehicles in this cluster:\n\n{combined_text}\n\nPlease analyze the trajectory clustering features shown in this image, combining this data. Describe the main paths, turning patterns, and characteristics. What type of driving behavior does this represent?'
            
            # Call API
            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=[
                    {
                        'role': 'system',
                        'content': [{'text': system_prompt}]
                    },
                    {
                        'role': 'user',
                        'content': [
                            {'image': f"data:image/png;base64,{base64_image}"},
                            {'text': user_prompt}
                        ]
                    }
                ]
            )
            
            # Extract analysis results and create individual HTML report
            if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                analysis = response.output.choices[0].message.content[0]['text']
                print(f"Analysis completed: {file_name}")
                
                # Create individual HTML report
                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Vehicle Trajectory Analysis - {file_name}</title>
                    <link rel="stylesheet" href="style.css">
                </head>
                <body>
                    <header>
                        <div class="container">
                            <h1>Vehicle Trajectory Analysis</h1>
                        </div>
                    </header>
                    
                    <div class="container">
                        <div class="meta-info">
                            <strong>File:</strong> {file_name}<br>
                            <strong>Analysis Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </div>
                """
                
                # Add vehicle ID and data information (if available)
                if vehicle_id and vehicle_text_data:
                    html_content += f"""
                        <div class="vehicle-data">
                            <h2>Vehicle ID: {vehicle_id}</h2>
                            <pre>{vehicle_text_data}</pre>
                        </div>
                    """
                elif isinstance(vehicle_data, dict) and len(vehicle_data) > 0 and all_vehicles_text_data:
                    # Handle cluster data (multiple vehicles)
                    html_content += f"""
                        <div class="vehicle-data">
                            <h2>Cluster Vehicle Data</h2>
                            <p>This cluster contains {len(vehicle_data)} vehicles</p>
                    """
                    
                    # Limit number of vehicles displayed to avoid overly large reports
                    display_limit = 3
                    displayed_count = 0
                    
                    for text_data in all_vehicles_text_data:
                        if displayed_count < display_limit:
                            html_content += f"""
                            <pre>{text_data}</pre>
                            """
                            displayed_count += 1
                        else:
                            break
                    
                    if len(all_vehicles_text_data) > display_limit:
                        html_content += f"""
                        <p>({len(all_vehicles_text_data) - display_limit} more vehicle data not shown)</p>
                        """
                    
                    html_content += "</div>"
                
                html_content += f"""
                        <div class="image-analysis">
                            <h2>Original Image</h2>
                            <img src="../{file_name}" alt="{file_name}">
                            
                            <div class="analysis-result">
                                <h2>Analysis Results</h2>
                                <pre>{analysis}</pre>
                            </div>
                        </div>
                        
                        <div class="nav-links">
                            <a href="index.html">Back to Index</a>
                        </div>
                    </div>
                    
                    <footer>
                        <div class="container">
                            <p>Generated by Vehicle Trajectory Analysis Tool</p>
                        </div>
                    </footer>
                </body>
                </html>
                """
                
                # Write HTML report
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                print(f"Report generated: {report_path}")
                
            else:
                print(f"Analysis failed: {file_name}, unexpected response format")
                print(f"Response content: {response}")
                
                # Create failure report
                html_content = f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Vehicle Trajectory Analysis - {file_name} (Failed)</title>
                    <link rel="stylesheet" href="style.css">
                </head>
                <body>
                    <header>
                        <div class="container">
                            <h1>Vehicle Trajectory Analysis (Failed)</h1>
                        </div>
                    </header>
                    
                    <div class="container">
                        <div class="meta-info">
                            <strong>File:</strong> {file_name}<br>
                            <strong>Analysis Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        </div>
                """
                
                # Add vehicle ID and data information (if available)
                if vehicle_id and vehicle_text_data:
                    html_content += f"""
                        <div class="vehicle-data">
                            <h2>Vehicle ID: {vehicle_id}</h2>
                            <pre>{vehicle_text_data}</pre>
                        </div>
                    """
                elif isinstance(vehicle_data, dict) and len(vehicle_data) > 0 and all_vehicles_text_data:
                    # Handle cluster data (multiple vehicles)
                    html_content += f"""
                        <div class="vehicle-data">
                            <h2>Cluster Vehicle Data</h2>
                            <p>This cluster contains {len(vehicle_data)} vehicles</p>
                    """
                    
                    # Limit number of vehicles displayed to avoid overly large reports
                    display_limit = 3
                    displayed_count = 0
                    
                    for text_data in all_vehicles_text_data:
                        if displayed_count < display_limit:
                            html_content += f"""
                            <pre>{text_data}</pre>
                            """
                            displayed_count += 1
                        else:
                            break
                    
                    if len(all_vehicles_text_data) > display_limit:
                        html_content += f"""
                        <p>({len(all_vehicles_text_data) - display_limit} more vehicle data not shown)</p>
                        """
                    
                    html_content += "</div>"
                
                html_content += f"""
                        <div class="image-analysis error-box">
                            <h2>Original Image</h2>
                            <img src="../{file_name}" alt="{file_name}">
                            
                            <div class="analysis-result">
                                <h2>Analysis Results</h2>
                                <pre>Analysis failed: Unexpected response format</pre>
                            </div>
                        </div>
                        
                        <div class="nav-links">
                            <a href="index.html">Back to Index</a>
                        </div>
                    </div>
                    
                    <footer>
                        <div class="container">
                            <p>Generated by Vehicle Trajectory Analysis Tool</p>
                        </div>
                    </footer>
                </body>
                </html>
                """
                
                # Write failure report
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                print(f"Failure report generated: {report_path}")
        
        except Exception as e:
            print(f"Error processing image: {file_name}, error: {e}")
            
            # Create error report
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Vehicle Trajectory Analysis - {file_name} (Error)</title>
                <link rel="stylesheet" href="style.css">
            </head>
            <body>
                <header>
                    <div class="container">
                        <h1>Vehicle Trajectory Analysis (Error)</h1>
                    </div>
                </header>
                
                <div class="container">
                    <div class="meta-info">
                        <strong>File:</strong> {file_name}<br>
                        <strong>Analysis Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    </div>
            """
            
            # Add vehicle ID and data information (if available)
            if vehicle_id and vehicle_text_data:
                html_content += f"""
                    <div class="vehicle-data">
                        <h2>Vehicle ID: {vehicle_id}</h2>
                        <pre>{vehicle_text_data}</pre>
                    </div>
                """
            elif isinstance(vehicle_data, dict) and len(vehicle_data) > 0 and all_vehicles_text_data:
                # Handle cluster data (multiple vehicles)
                html_content += f"""
                    <div class="vehicle-data">
                        <h2>Cluster Vehicle Data</h2>
                        <p>This cluster contains {len(vehicle_data)} vehicles</p>
                """
                
                # Limit number of vehicles displayed to avoid overly large reports
                display_limit = 3
                displayed_count = 0
                
                for text_data in all_vehicles_text_data:
                    if displayed_count < display_limit:
                        html_content += f"""
                        <pre>{text_data}</pre>
                        """
                        displayed_count += 1
                    else:
                        break
                
                if len(all_vehicles_text_data) > display_limit:
                    html_content += f"""
                    <p>({len(all_vehicles_text_data) - display_limit} more vehicle data not shown)</p>
                    """
                
                html_content += "</div>"
            
            html_content += f"""
                    <div class="image-analysis error-box">
                        <h2>Original Image</h2>
                        <img src="../{file_name}" alt="{file_name}">
                        
                        <div class="analysis-result">
                            <h2>Analysis Results</h2>
                            <pre>Processing error: {str(e)}</pre>
                        </div>
                    </div>
                    
                    <div class="nav-links">
                        <a href="index.html">Back to Index</a>
                    </div>
                </div>
                
                <footer>
                    <div class="container">
                        <p>Generated by Vehicle Trajectory Analysis Tool</p>
                    </div>
                </footer>
            </body>
            </html>
            """
            
            # Write error report
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            print(f"Error report generated: {report_path}")
    
    # Create index page
    index_path = os.path.join(reports_folder, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Vehicle Trajectory Analysis Reports</title>
            <link rel="stylesheet" href="style.css">
        </head>
        <body>
            <header>
                <div class="container">
                    <h1>Vehicle Trajectory Analysis Reports</h1>
                </div>
            </header>
            
            <div class="container">
                <div class="meta-info">
                    <strong>Generation Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                    <strong>Total Images Analyzed:</strong> {len(png_files)}
                </div>
                
                <div class="image-analysis">
                    <h2>Report Index</h2>
                    <ul class="report-list">
        """)
        
        # Add links to each report
        for png_file in png_files:
            file_name = os.path.basename(png_file)
            file_base_name = os.path.splitext(file_name)[0]
            report_filename = f"{file_base_name}_analysis.html"
            
            f.write(f'''
                        <li class="report-item">
                            <a href="{report_filename}">{file_name}</a>
                        </li>
            ''')
        
        f.write("""
                    </ul>
                </div>
            </div>
            
            <footer>
                <div class="container">
                    <p>Generated by Vehicle Trajectory Analysis Tool</p>
                </div>
            </footer>
        </body>
        </html>
        """)
    
    print(f"Index page generated: {index_path}")
    print(f"All analysis reports saved to folder: {reports_folder}")

if __name__ == "__main__":
    folder_path = "single_vehicle_clusters/"
    analyze_images(folder_path)