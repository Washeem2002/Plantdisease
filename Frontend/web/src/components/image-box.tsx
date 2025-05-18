"use client"
import { useState, ChangeEvent, FormEvent, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"
import Image from "next/image"
import LeafSVG from "@/components/assets/Leaf"
import { ReloadIcon } from "@radix-ui/react-icons"
import Result from "@/components/result"

export function ImageBox() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imageURL, setImageURL] = useState<string>()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const { toast } = useToast()

  function onImageUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    setImageFile(file)
    setImageURL(URL.createObjectURL(file))

    toast({
      variant: "success",
      title: "Image Uploaded",
      description: `${file.name} uploaded successfully`,
    })
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (!imageFile) return

    const formData = new FormData()
    formData.append("file", imageFile)

    setLoading(true)
    setResult(null)

    try {
      // Step 1: POST to /predict
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) throw new Error("Prediction failed")
      const predictionData = await res.json()
      console.log("Prediction Data:", predictionData)

      // Extract disease name safely
      const diseaseName = predictionData;
      console.log("Extracted Disease Name:", diseaseName)

      let cureInfo = {}

      // Step 2: Call /get_cure if diseaseName exists
      if (diseaseName) {
        console.log("Calling /get_cure with disease:", diseaseName)
        const cureRes = await fetch("http://127.0.0.1:5000/get_cure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ disease: diseaseName }),
        })

        if (!cureRes.ok) throw new Error("Failed to fetch cure info")

        cureInfo = await cureRes.json()
        console.log("Cure Info:", cureInfo)
      } else {
        console.warn("No disease name found, skipping /get_cure call.")
      }

      // Step 3: Merge and display result
      setResult({
        ...predictionData,
        cure_info: cureInfo,
      })

      toast({
        title: "Prediction & Cure fetched",
        description: "Check the result below",
      })
    } catch (err) {
      console.error(err)
      toast({
        variant: "destructive",
        title: "Error",
        description: (err as Error).message,
      })
    } finally {
      setLoading(false)
    }
  }
  useEffect(()=>{
    console.log(result);
  },[result])

  return (
    <section className="mt-8 md:mt-4">
      <form encType="multipart/form-data" method="post" onSubmit={handleSubmit}>
        <div className="flex flex-col items-center">
          <label htmlFor="plant-image" className="cursor-pointer">
            <div className="relative w-72 mt-4 flex items-center justify-center aspect-square mx-auto border-2 dark:border-white border-black border-dashed rounded-lg">
              {imageURL ? (
                <Image
                  src={imageURL}
                  alt="Plant"
                  fill
                  className="rounded-lg object-cover"
                />
              ) : (
                <div className="flex flex-col gap-2 p-4 justify-center items-center">
                  <LeafSVG />
                  <p className="text-center">Upload Plant Image Here</p>
                </div>
              )}
              <input
                type="file"
                name="plant-image"
                id="plant-image"
                className="hidden"
                accept=".png, .jpeg, .jpg"
                onChange={onImageUpload}
                required
              />
            </div>
          </label>

          <div className="mt-4">
            {!imageFile ? (
              <Button disabled>Add Image to Proceed</Button>
            ) : (
              <div className="flex flex-col gap-4 items-center">
                <p>{imageFile.name} Uploaded!</p>
                <Button type="submit" disabled={loading}>
                  {loading && (
                    <ReloadIcon className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Detect Disease
                </Button>
              </div>
            )}
          </div>
        </div>
      </form>

      {/* Show results if available */}
         {result && <Result data={result} />}   
    </section>
  )
}
