/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ['shared'],
  output: 'standalone',
}

export default nextConfig
