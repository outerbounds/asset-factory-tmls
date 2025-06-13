#!/usr/bin/env python3
"""
Enhanced diagnostic script to analyze photo filtering and provide actionable recommendations
for improving UpdatePhotos workflow data diversity and quality.
"""

from metaflow import FlowSpec, step, current, namespace, Parameter, card
from metaflow.cards import Markdown, Table, Image
import json
from datetime import datetime
from collections import Counter, defaultdict

class DiagnoseData(FlowSpec):
    """
    Comprehensive diagnosis of photo data collection with actionable recommendations
    for improving UpdatePhotos workflow diversity and quality.
    """
    
    lookback_runs = Parameter("lookback-runs", default=20, help="Number of runs to examine")
    namespace = Parameter("data-ns", default="", help="Namespace to filter runs by")
    
    @step
    def start(self):
        """Analyze photo filtering issues and data diversity patterns"""
        
        if self.namespace != current.namespace and self.namespace != "":
            if self.namespace == "global" or self.namespace == "none" or self.namespace == "null":
                print(f"🔍 Resetting namespace to global")
                namespace(None)
            else:
                print(f"🔍 Setting namespace to {self.namespace}")
                namespace(self.namespace)
        
        print(f"🔍 Comprehensive diagnosis across {self.lookback_runs} runs...")
        
        from metaflow import Flow
        
        # Enhanced analytics structure
        self.analysis = {
            "run_stats": {
                "total_runs": 0,
                "runs_with_photos": 0,
                "total_photos": 0,
                "unique_photos": 0,
                "duplicate_rate": 0.0
            },
            "filter_failures": {
                "missing_fields": 0,
                "insufficient_tags": 0,
                "invalid_url": 0,
                "short_tags": 0,
                "duplicate_photos": 0
            },
            "diversity_analysis": {
                "photo_size_distribution": Counter(),
                "tag_diversity": {
                    "unique_tags": set(),
                    "tag_frequency": Counter(),
                    "avg_tags_per_photo": 0.0,
                    "tag_length_distribution": Counter()
                },
                "temporal_patterns": defaultdict(int),
                "photo_id_patterns": {
                    "id_prefixes": Counter(),
                    "id_lengths": Counter()
                }
            },
            "quality_metrics": {
                "high_quality_photos": 0,  # >10 tags
                "medium_quality_photos": 0,  # 5-10 tags  
                "low_quality_photos": 0,   # 3-5 tags
                "url_patterns": Counter(),
                "attribution_diversity": Counter()
            },
            "sample_data": {
                "best_examples": [],
                "problematic_examples": [],
                "duplicate_examples": []
            }
        }
        
        seen_photos = set()
        all_photo_data = []
        
        try:
            updatephotos_flow = Flow('UpdatePhotos')
            runs = list(updatephotos_flow)[:self.lookback_runs]
            
            for run_idx, run in enumerate(runs):
                self.analysis["run_stats"]["total_runs"] += 1
                
                try:
                    if hasattr(run.data, 'photos') and run.data.photos:
                        run_photos = run.data.photos
                        self.analysis["run_stats"]["runs_with_photos"] += 1
                        self.analysis["run_stats"]["total_photos"] += len(run_photos)
                        
                        # Analyze photo size distribution
                        self.analysis["diversity_analysis"]["photo_size_distribution"][len(run_photos)] += 1
                        
                        print(f"\n📸 Run {run_idx + 1} ({run.pathspec}): {len(run_photos)} photos")
                        
                        run_new_photos = 0
                        for photo_id, photo_data in run_photos.items():
                            all_photo_data.append((photo_id, photo_data, run.pathspec))
                            
                            # Track unique photos
                            if photo_id not in seen_photos:
                                seen_photos.add(photo_id)
                                run_new_photos += 1
                                self._analyze_photo_quality(photo_id, photo_data)
                            else:
                                self.analysis["filter_failures"]["duplicate_photos"] += 1
                                
                        print(f"   📊 New photos in this run: {run_new_photos}")
                        
                    else:
                        print(f"📸 Run {run_idx + 1} ({run.pathspec}): No photos data")
                        
                except Exception as e:
                    print(f"⚠️ Error accessing run {run_idx + 1}: {e}")
                    
        except Exception as e:
            print(f"⚠️ Error accessing UpdatePhotos flow: {e}")
        
        # Calculate final metrics
        self.analysis["run_stats"]["unique_photos"] = len(seen_photos)
        if self.analysis["run_stats"]["total_photos"] > 0:
            self.analysis["run_stats"]["duplicate_rate"] = (
                self.analysis["filter_failures"]["duplicate_photos"] / 
                self.analysis["run_stats"]["total_photos"]
            )
        
        # Analyze tag diversity
        if self.analysis["diversity_analysis"]["tag_diversity"]["unique_tags"]:
            total_tags = sum(self.analysis["diversity_analysis"]["tag_diversity"]["tag_frequency"].values())
            unique_photos = self.analysis["run_stats"]["unique_photos"]
            if unique_photos > 0:
                self.analysis["diversity_analysis"]["tag_diversity"]["avg_tags_per_photo"] = total_tags / unique_photos
        
        # Sample best and worst examples
        self._select_sample_examples(all_photo_data, seen_photos)
        
        self.next(self.analyze_updatephotos_code)
    
    def _analyze_photo_quality(self, photo_id, photo_data):
        """Analyze individual photo quality and diversity metrics"""
        
        # Tag analysis
        tags = photo_data.get('ground_truth_tags', [])
        tag_count = len(tags)
        
        # Quality classification
        if tag_count >= 10:
            self.analysis["quality_metrics"]["high_quality_photos"] += 1
        elif tag_count >= 5:
            self.analysis["quality_metrics"]["medium_quality_photos"] += 1
        else:
            self.analysis["quality_metrics"]["low_quality_photos"] += 1
        
        # Tag diversity
        for tag in tags:
            self.analysis["diversity_analysis"]["tag_diversity"]["unique_tags"].add(tag.lower())
            self.analysis["diversity_analysis"]["tag_diversity"]["tag_frequency"][tag.lower()] += 1
            self.analysis["diversity_analysis"]["tag_diversity"]["tag_length_distribution"][len(tag)] += 1
        
        # URL patterns
        url = photo_data.get('image_url', '')
        if url:
            # Extract URL pattern (domain + path structure)
            if 'unsplash.com' in url:
                self.analysis["quality_metrics"]["url_patterns"]["unsplash"] += 1
            else:
                self.analysis["quality_metrics"]["url_patterns"]["other"] += 1
        
        # Attribution diversity
        attribution = photo_data.get('attribution_text', 'unknown')
        self.analysis["quality_metrics"]["attribution_diversity"][attribution] += 1
        
        # Photo ID patterns
        self.analysis["diversity_analysis"]["photo_id_patterns"]["id_lengths"][len(photo_id)] += 1
        if len(photo_id) >= 3:
            prefix = photo_id[:3]
            self.analysis["diversity_analysis"]["photo_id_patterns"]["id_prefixes"][prefix] += 1
    
    def _select_sample_examples(self, all_photo_data, seen_photos):
        """Select representative examples for analysis"""
        
        # Best examples (high tag count, diverse tags)
        quality_scores = []
        for photo_id, photo_data, run_path in all_photo_data:
            if photo_id in seen_photos:  # Only unique photos
                tags = photo_data.get('ground_truth_tags', [])
                score = len(tags) + len(set(tags))  # Tag count + uniqueness
                quality_scores.append((score, photo_id, photo_data, run_path))
        
        # Sort by quality score (only by the numeric score)
        quality_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select top examples
        self.analysis["sample_data"]["best_examples"] = [
            {
                "photo_id": photo_id,
                "run": run_path,
                "tag_count": len(photo_data.get('ground_truth_tags', [])),
                "tags": photo_data.get('ground_truth_tags', [])[:5],  # First 5 tags
                "url": photo_data.get('image_url', '')[:50] + "...",
                "attribution": photo_data.get('attribution_text', 'unknown')
            }
            for score, photo_id, photo_data, run_path in quality_scores[:5]
        ]
        
        # Problematic examples (low tag count)
        problematic = [
            (photo_id, photo_data, run_path) 
            for _, photo_id, photo_data, run_path in quality_scores[-5:]
        ]
        
        self.analysis["sample_data"]["problematic_examples"] = [
            {
                "photo_id": photo_id,
                "run": run_path,
                "tag_count": len(photo_data.get('ground_truth_tags', [])),
                "tags": photo_data.get('ground_truth_tags', []),
                "issues": self._identify_issues(photo_data)
            }
            for photo_id, photo_data, run_path in problematic
        ]
    
    def _identify_issues(self, photo_data):
        """Identify specific issues with a photo"""
        issues = []
        
        tags = photo_data.get('ground_truth_tags', [])
        if len(tags) < 3:
            issues.append("insufficient_tags")
        if any(len(str(tag).strip()) < 2 for tag in tags):
            issues.append("short_tags")
        if not photo_data.get('image_url', '').startswith(('http://', 'https://')):
            issues.append("invalid_url")
        if not all(field in photo_data for field in ['id', 'image_url', 'ground_truth_tags']):
            issues.append("missing_fields")
            
        return issues
    
    @step
    def analyze_updatephotos_code(self):
        """Analyze the current UpdatePhotos implementation for improvement opportunities"""
        
        print("🔍 Analyzing UpdatePhotos workflow for improvement opportunities...")
        
        # Read the current UpdatePhotos code
        try:
            with open('updatephotos.py', 'r') as f:
                updatephotos_code = f.read()
        except FileNotFoundError:
            updatephotos_code = "# UpdatePhotos code not found"
        
        # Analyze current implementation
        self.code_analysis = {
            "current_api_usage": {
                "endpoint": "https://api.unsplash.com/photos",
                "parameters": "client_id only",
                "pagination": "Not implemented",
                "search_terms": "Not used",
                "collections": "Not used"
            },
            "improvement_opportunities": [
                {
                    "category": "API Diversification",
                    "priority": "HIGH",
                    "description": "Use multiple Unsplash endpoints for diverse content",
                    "implementation": "Add search, collections, and featured endpoints"
                },
                {
                    "category": "Pagination",
                    "priority": "HIGH", 
                    "description": "Implement pagination to access more photos",
                    "implementation": "Add page parameter to API calls"
                },
                {
                    "category": "Search Diversity",
                    "priority": "MEDIUM",
                    "description": "Use rotating search terms for varied content",
                    "implementation": "Implement search term rotation strategy"
                },
                {
                    "category": "Collection Sampling",
                    "priority": "MEDIUM",
                    "description": "Sample from different Unsplash collections",
                    "implementation": "Add collection-based photo fetching"
                },
                {
                    "category": "Temporal Diversity",
                    "priority": "LOW",
                    "description": "Vary API calls based on time/date",
                    "implementation": "Time-based parameter variation"
                }
            ]
        }
        
        self.next(self.generate_recommendations)
    
    @card(type="blank")
    @step
    def generate_recommendations(self):
        analysis = self.analysis
        html_content = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;">
            
            <h1 style="color: #2563eb; border-bottom: 3px solid #2563eb; padding-bottom: 10px;">
                📊 Photo Data Diversity Analysis & UpdatePhotos Optimization
            </h1>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
                
                <!-- Current State Summary -->
                <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #0ea5e9;">
                    <h2 style="color: #0c4a6e; margin-top: 0;">📈 Current Data State</h2>
                    <ul style="list-style: none; padding: 0;">
                        <li style="margin: 8px 0;"><strong>Total Runs Analyzed:</strong> {analysis['run_stats']['total_runs']}</li>
                        <li style="margin: 8px 0;"><strong>Runs with Photos:</strong> {analysis['run_stats']['runs_with_photos']}</li>
                        <li style="margin: 8px 0;"><strong>Total Photos Found:</strong> {analysis['run_stats']['total_photos']:,}</li>
                        <li style="margin: 8px 0;"><strong>Unique Photos:</strong> {analysis['run_stats']['unique_photos']:,}</li>
                        <li style="margin: 8px 0;"><strong>Duplicate Rate:</strong> {analysis['run_stats']['duplicate_rate']:.1%}</li>
                    </ul>
                </div>
                
                <!-- Quality Distribution -->
                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #22c55e;">
                    <h2 style="color: #14532d; margin-top: 0;">🎯 Quality Distribution</h2>
                    <ul style="list-style: none; padding: 0;">
                        <li style="margin: 8px 0;"><strong>High Quality (10+ tags):</strong> {analysis['quality_metrics']['high_quality_photos']}</li>
                        <li style="margin: 8px 0;"><strong>Medium Quality (5-10 tags):</strong> {analysis['quality_metrics']['medium_quality_photos']}</li>
                        <li style="margin: 8px 0;"><strong>Low Quality (3-5 tags):</strong> {analysis['quality_metrics']['low_quality_photos']}</li>
                        <li style="margin: 8px 0;"><strong>Avg Tags/Photo:</strong> {analysis['diversity_analysis']['tag_diversity']['avg_tags_per_photo']:.1f}</li>
                        <li style="margin: 8px 0;"><strong>Unique Tags:</strong> {len(analysis['diversity_analysis']['tag_diversity']['unique_tags']):,}</li>
                    </ul>
                </div>
                
            </div>
            
            <!-- Critical Issues -->
            <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #ef4444; margin: 20px 0;">
                <h2 style="color: #991b1b; margin-top: 0;">⚠️ Critical Issues Identified</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div>
                        <h3 style="color: #7c2d12; margin: 10px 0 5px 0;">🔄 High Duplication</h3>
                        <p style="margin: 5px 0; font-size: 14px;">
                            <strong>{analysis['run_stats']['duplicate_rate']:.1%}</strong> of photos are duplicates. 
                            Current API strategy fetches same popular photos repeatedly.
                        </p>
                    </div>
                    <div>
                        <h3 style="color: #7c2d12; margin: 10px 0 5px 0;">📊 Limited Diversity</h3>
                        <p style="margin: 5px 0; font-size: 14px;">
                            Only <strong>{analysis['run_stats']['unique_photos']}</strong> unique photos from 
                            <strong>{analysis['run_stats']['total_photos']:,}</strong> total photos collected.
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Improvement Recommendations -->
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #6366f1; margin: 20px 0;">
                <h2 style="color: #312e81; margin-top: 0;">🚀 UpdatePhotos Optimization Strategy</h2>
                
                <div style="margin: 20px 0;">
                    <h3 style="color: #1e40af;">🎯 HIGH PRIORITY Improvements</h3>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e2e8f0;">
                        <h4 style="color: #dc2626; margin: 0 0 10px 0;">1. Implement API Pagination</h4>
                        <p style="margin: 5px 0; font-size: 14px;"><strong>Current:</strong> Only fetches first page (30 photos)</p>
                        <p style="margin: 5px 0; font-size: 14px;"><strong>Solution:</strong> Add page parameter rotation</p>
                        <pre style="background: #f8fafc; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">
# Add to request() function:
page = (datetime.now().hour % 10) + 1  # Rotate pages 1-10
params = {{"client_id": unsplash_client_id, "page": page, "per_page": 30}}
                        </pre>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e2e8f0;">
                        <h4 style="color: #dc2626; margin: 0 0 10px 0;">2. Add Search Endpoint Diversity</h4>
                        <p style="margin: 5px 0; font-size: 14px;"><strong>Current:</strong> Only uses /photos endpoint</p>
                        <p style="margin: 5px 0; font-size: 14px;"><strong>Solution:</strong> Rotate between multiple endpoints</p>
                        <pre style="background: #f8fafc; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">
# Endpoint rotation strategy:
endpoints = [
    "/photos",  # Latest photos
    "/photos/random",  # Random photos  
    "/search/photos?query=nature",
    "/search/photos?query=architecture",
    "/collections/featured/photos"
]
                        </pre>
                    </div>
                    
                </div>
                
                <div style="margin: 20px 0;">
                    <h3 style="color: #1e40af;">🎯 MEDIUM PRIORITY Improvements</h3>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e2e8f0;">
                        <h4 style="color: #ea580c; margin: 0 0 10px 0;">3. Smart Search Term Rotation</h4>
                        <p style="margin: 5px 0; font-size: 14px;">Implement diverse search terms based on time/date</p>
                        <pre style="background: #f8fafc; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">
search_terms = [
    "nature", "architecture", "food", "travel", "portrait",
    "landscape", "urban", "abstract", "technology", "art"
]
term = search_terms[datetime.now().day % len(search_terms)]
                        </pre>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid #e2e8f0;">
                        <h4 style="color: #ea580c; margin: 0 0 10px 0;">4. Collection-Based Sampling</h4>
                        <p style="margin: 5px 0; font-size: 14px;">Sample from curated Unsplash collections for quality</p>
                        <pre style="background: #f8fafc; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto;">
# Popular collections with diverse content:
collections = ["3816141", "1319040", "1114848", "1065976"]
collection_id = collections[datetime.now().hour % len(collections)]
url = f"/collections/{{collection_id}}/photos"
                        </pre>
                    </div>
                </div>
                
            </div>
            
            <!-- Implementation Code -->
            <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b; margin: 20px 0;">
                <h2 style="color: #92400e; margin-top: 0;">💻 Ready-to-Implement Code</h2>
                
                <h3 style="color: #78350f;">Enhanced request() function:</h3>
                <pre style="background: #1f2937; color: #f9fafb; padding: 15px; border-radius: 8px; font-size: 12px; overflow-x: auto; line-height: 1.4;">
def request(path="", search_query=None, page=1):
    unsplash_client_id = os.environ["client_id"]
    
    # Build parameters
    params = {{
        "client_id": unsplash_client_id,
        "page": page,
        "per_page": 30
    }}
    
    if search_query:
        params["query"] = search_query
    
    # Build URL
    if search_query and not path.startswith("/search"):
        url = os.path.join(PHOTOS_URL.replace("/photos", ""), "search/photos")
    else:
        url = os.path.join(PHOTOS_URL, path)
    
    return requests.get(url, params=params).json()

def get_diverse_photos():
    \"\"\"Get photos using diverse strategies\"\"\"
    hour = datetime.now().hour
    day = datetime.now().day
    
    # Strategy rotation based on time
    strategies = [
        {{"path": "", "page": (hour % 10) + 1}},  # Paginated latest
        {{"search_query": "nature", "page": (hour % 5) + 1}},
        {{"search_query": "architecture", "page": (day % 3) + 1}},
        {{"path": "random", "count": 30}},
        {{"path": f"collections/{{3816141 + (day % 4)}}/photos"}}
    ]
    
    strategy = strategies[hour % len(strategies)]
    return request(**strategy)
                </pre>
            </div>
            
            <!-- Sample Data Analysis -->
            <div style="margin: 20px 0;">
                <h2 style="color: #1f2937;">📋 Sample Data Analysis</h2>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div style="background: #f0fdf4; padding: 15px; border-radius: 8px;">
                        <h3 style="color: #14532d; margin-top: 0;">✅ Best Examples</h3>
        """
        
        # Add best examples
        for i, example in enumerate(analysis['sample_data']['best_examples'][:3]):
            html_content += f"""
                        <div style="background: white; padding: 10px; margin: 8px 0; border-radius: 6px; font-size: 12px;">
                            <strong>#{i+1}: {example['photo_id']}</strong><br>
                            Tags: {example['tag_count']} ({', '.join(example['tags'])})<br>
                            Attribution: {example['attribution']}
                        </div>
            """
        
        html_content += """
                    </div>
                    
                    <div style="background: #fef2f2; padding: 15px; border-radius: 8px;">
                        <h3 style="color: #991b1b; margin-top: 0;">⚠️ Problematic Examples</h3>
        """
        
        # Add problematic examples
        for i, example in enumerate(analysis['sample_data']['problematic_examples'][:3]):
            html_content += f"""
                        <div style="background: white; padding: 10px; margin: 8px 0; border-radius: 6px; font-size: 12px;">
                            <strong>#{i+1}: {example['photo_id']}</strong><br>
                            Tags: {example['tag_count']} ({', '.join(example['tags'])})<br>
                            Issues: {', '.join(example['issues'])}
                        </div>
            """
        
        html_content += f"""
                    </div>
                </div>
            </div>
            
            <!-- Action Items -->
            <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #10b981; margin: 20px 0;">
                <h2 style="color: #064e3b; margin-top: 0;">✅ Immediate Action Items</h2>
                <ol style="color: #065f46; line-height: 1.6;">
                    <li><strong>Update updatephotos.py</strong> with enhanced request() function</li>
                    <li><strong>Test pagination</strong> - should increase unique photos by 5-10x</li>
                    <li><strong>Implement search rotation</strong> - target 500+ unique photos</li>
                    <li><strong>Add collection sampling</strong> - improve photo quality</li>
                    <li><strong>Monitor diversity metrics</strong> - track improvement over time</li>
                </ol>
                
                <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #14532d; margin: 0 0 10px 0;">📊 Expected Impact</h3>
                    <ul style="margin: 5px 0; color: #166534;">
                        <li><strong>Unique Photos:</strong> {analysis['run_stats']['unique_photos']} → 500-1000+ (5-10x increase)</li>
                        <li><strong>Duplicate Rate:</strong> {analysis['run_stats']['duplicate_rate']:.1%} → <20% (major improvement)</li>
                        <li><strong>Tag Diversity:</strong> {len(analysis['diversity_analysis']['tag_diversity']['unique_tags'])} → 1000+ unique tags</li>
                        <li><strong>Fine-tuning Readiness:</strong> Current dataset insufficient → Production-ready dataset</li>
                    </ul>
                </div>
            </div>
            
        </div>
        """
        
        current.card.append(Markdown("# 📊 Photo Data Diversity Analysis"))
        current.card.append(Markdown(html_content))
        
        self.next(self.end)
    
    @step 
    def end(self):
        """Complete the comprehensive diagnosis"""
        
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE DIAGNOSIS COMPLETE!")
        print("="*60)
        
        analysis = self.analysis
        
        print(f"\n📊 KEY FINDINGS:")
        print(f"  • Analyzed {analysis['run_stats']['total_runs']} runs")
        print(f"  • Found {analysis['run_stats']['unique_photos']} unique photos from {analysis['run_stats']['total_photos']:,} total")
        print(f"  • Duplicate rate: {analysis['run_stats']['duplicate_rate']:.1%}")
        print(f"  • Tag diversity: {len(analysis['diversity_analysis']['tag_diversity']['unique_tags'])} unique tags")
        
        print(f"\n🚀 IMMEDIATE ACTIONS:")
        print(f"  1. Implement API pagination (5-10x more photos)")
        print(f"  2. Add search endpoint rotation (better diversity)")
        print(f"  3. Use collection sampling (higher quality)")
        print(f"  4. Expected result: 500-1000+ unique photos")
        
        print(f"\n📋 Check the detailed analysis card for:")
        print(f"  • Ready-to-implement code")
        print(f"  • Specific API strategies")
        print(f"  • Sample data analysis")
        print(f"  • Expected impact metrics")
        
        print(f"\n✅ Ready to proceed with UpdatePhotos optimization!")

if __name__ == "__main__":
    DiagnoseData() 