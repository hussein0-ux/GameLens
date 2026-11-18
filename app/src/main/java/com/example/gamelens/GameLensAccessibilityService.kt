package com.example.gamelens

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.view.accessibility.AccessibilityEvent

/**
 * Does no accessibility work itself. It exists only so the enhanced-picture overlay can be created
 * with TYPE_ACCESSIBILITY_OVERLAY, which Android treats as a trusted overlay: touches pass through
 * it to the game below instead of being blocked like they are for normal "draw over other apps"
 * overlays.
 */
class GameLensAccessibilityService : AccessibilityService() {

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {}

    override fun onUnbind(intent: Intent?): Boolean {
        if (instance === this) instance = null
        return super.onUnbind(intent)
    }

    companion object {
        @Volatile var instance: GameLensAccessibilityService? = null
    }
}
