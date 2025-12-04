class AlertSystem:
    """Unified alert system for drowsiness and distraction"""
    
    def __init__(self):
        self.alert_history = []
        self.max_history = 100
        
    def evaluate(self, drowsy_status, drowsy_severity, distraction_activity, distraction_severity):
        """
        Evaluate combined status and generate alert
        Returns: (alert_level, alert_message, combined_severity)
        
        alert_level: 'none', 'low', 'medium', 'high', 'critical'
        """
        combined_severity = max(drowsy_severity, distraction_severity)
        alert_level = 'none'
        messages = []
        
        # Critical alerts (severity >= 9)
        if drowsy_severity >= 9:
            alert_level = 'critical'
            messages.append("🚨 DROWSINESS DETECTED - PULL OVER IMMEDIATELY!")
        
        if distraction_activity == 'drinking' and distraction_severity >= 8:
            alert_level = 'critical'
            messages.append("🚨 DRINKING WHILE DRIVING - STOP THE VEHICLE!")
        
        # High alerts (severity 7-8)
        if drowsy_severity >= 7 and alert_level != 'critical':
            alert_level = 'high'
            messages.append("⚠️ High drowsiness level detected")
        
        if distraction_activity == 'using_phone' and alert_level != 'critical':
            alert_level = 'high'
            messages.append("⚠️ Phone usage detected - Keep eyes on road!")
        
        # Medium alerts (severity 4-6)
        if distraction_activity == 'turning' and alert_level == 'none':
            alert_level = 'medium'
            messages.append("⚠️ Turning detected - Brief attention shift")
        
        if distraction_activity == 'hair_makeup' and alert_level not in ['critical', 'high']:
            alert_level = 'medium'
            messages.append("⚠️ Grooming activity detected")
        
        if drowsy_severity >= 4 and alert_level == 'none':
            alert_level = 'medium'
            messages.append("⚠️ Signs of drowsiness - Stay alert")
        
        # Low alerts (severity 1-3)
        if distraction_activity == 'radio' and alert_level == 'none':
            alert_level = 'low'
            messages.append("ℹ️ Radio adjustment detected")
        
        if distraction_activity == 'others_activities' and alert_level == 'none':
            alert_level = 'low'
            messages.append("ℹ️ Unusual activity detected")
        
        # Safe status
        if alert_level == 'none' and drowsy_status == 'safe' and distraction_activity == 'safe_driving':
            messages.append("✅ Safe driving - All systems normal")
        
        alert_message = " | ".join(messages) if messages else "Monitoring..."
        
        # Store in history
        self.alert_history.append({
            'drowsy_status': drowsy_status,
            'drowsy_severity': drowsy_severity,
            'distraction': distraction_activity,
            'distraction_severity': distraction_severity,
            'alert_level': alert_level,
            'combined_severity': combined_severity
        })
        
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        return alert_level, alert_message, combined_severity
    
    def get_statistics(self):
        """Get summary statistics from alert history"""
        if not self.alert_history:
            return {
                'total_events': 0,
                'critical_events': 0,
                'high_events': 0,
                'medium_events': 0,
                'low_events': 0,
                'safe_percentage': 0,
                'avg_severity': 0
            }
        
        total = len(self.alert_history)
        critical = sum(1 for a in self.alert_history if a['alert_level'] == 'critical')
        high = sum(1 for a in self.alert_history if a['alert_level'] == 'high')
        medium = sum(1 for a in self.alert_history if a['alert_level'] == 'medium')
        low = sum(1 for a in self.alert_history if a['alert_level'] == 'low')
        safe = sum(1 for a in self.alert_history if a['alert_level'] == 'none')
        
        avg_severity = sum(a['combined_severity'] for a in self.alert_history) / total
        
        return {
            'total_events': total,
            'critical_events': critical,
            'high_events': high,
            'medium_events': medium,
            'low_events': low,
            'safe_percentage': (safe / total) * 100,
            'avg_severity': avg_severity
        }
    
    def get_color_for_level(self, alert_level):
        """Get color code for alert level"""
        colors = {
            'none': '#28a745',      # Green
            'low': '#17a2b8',       # Blue
            'medium': '#ffc107',    # Yellow
            'high': '#fd7e14',      # Orange
            'critical': '#dc3545'   # Red
        }
        return colors.get(alert_level, '#6c757d')
    
    def reset(self):
        """Clear alert history"""
        self.alert_history = []
