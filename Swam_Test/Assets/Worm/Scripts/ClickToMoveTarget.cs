using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace SwarmTest
{
    public class ClickToMoveTarget : MonoBehaviour
    {
        public Camera cam;
        public LayerMask mask = ~0;
        public float heightOffset = 0.05f;

        void Update()
        {
            bool held;
            Vector2 mouse;
#if ENABLE_INPUT_SYSTEM
            if (Mouse.current == null) return;
            held = Mouse.current.leftButton.isPressed;
            mouse = Mouse.current.position.ReadValue();
#else
            held = Input.GetMouseButton(0);
            mouse = Input.mousePosition;
#endif
            if (!held) return;

            RaycastHit hit;
            if (Physics.Raycast(cam.ScreenPointToRay(mouse), out hit, 500f, mask, QueryTriggerInteraction.Ignore))
                transform.position = hit.point + hit.normal * heightOffset;
        }
    }
}
