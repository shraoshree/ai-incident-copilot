import { defineConfig } from "vite";

const proxy = { "/incident": "http://127.0.0.1:8000" };

export default defineConfig({ server: { proxy }, preview: { proxy } });
